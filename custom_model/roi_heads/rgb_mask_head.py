# Code partially taken from detectron2/modeling/roi_heads/roi_mask_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

from .utils import cosine_similarity_matrix, normalize_adjacency_matrix, create_graph_data


RGB_MASK_HEAD_REGISTRY = Registry("RGB Mask Head")
RGB_MASK_HEAD_REGISTRY.__doc__ = "Registry for RGB Mask Heads"


class BaseMaskRCNNHead(nn.Module):
    @configurable
    def __init__(self, *, loss_weight: float = 1.0, vis_period: int = 0):
        super().__init__()
        self.vis_period = vis_period
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": 0}


@RGB_MASK_HEAD_REGISTRY.register()
class ClassicMaskRCNNConvUpsampleHead(BaseMaskRCNNHead, nn.Sequential):
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = nn.ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.deconv_relu = nn.ReLU()
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_RGB_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_RGB_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_RGB_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_RGB_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.DATASETS.NUM_CLASSES
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x
    
    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        mask_features = x
        x = self.deconv_relu(self.deconv(mask_features))
        mask_pred = self.predictor(x)
        return mask_pred, mask_features


class CNN(nn.Module):
    def __init__(self, input_channels, channels, norm, num_layers):
        super(CNN, self).__init__()
        self.num_layers = num_layers
        self.conv_norm_relus = []
        # Première couche avec input_channels
        conv = Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not norm, norm=get_norm(norm, channels), activation=nn.ReLU())
        self.add_module("mask_fcn1", conv)
        self.conv_norm_relus.append(conv)
        # Couches suivantes
        for i in range(1, num_layers):
            conv = Conv2d(channels, channels, kernel_size=3, padding=1, bias=not norm, norm=get_norm(norm, channels), activation=nn.ReLU())
            self.add_module("mask_fcn{}".format(i + 1), conv)
            self.conv_norm_relus.append(conv)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
    
    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        return x

class GCN(nn.Module):
    def __init__(self, channels, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.gelu = nn.GELU()
        for _ in range(num_layers):
            self.convs.append(GCNConv(channels, channels))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.gelu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, num_classes, norm, num_layers):
        super(Upsample, self).__init__()
        self.num_layers = num_layers
        self.conv_norm_relus = []
        for i in range(num_layers):
            conv = Conv2d(channels, channels, kernel_size=3, padding=1, bias=not norm, norm=get_norm(norm, channels), activation=nn.ReLU())
            self.add_module("mask_fcn_upsample{}".format(i + 1), conv)
            self.conv_norm_relus.append(conv)
        self.deconv = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.deconv_activation = nn.ReLU()
        self.predictor = Conv2d(channels, num_classes, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        weight_init.c2_msra_fill(self.deconv)
        nn.init.normal_(self.predictor.weight, std=0.001) # Use it reduces the training performance, but makes it more stable
        #weight_init.c2_msra_fill(self.predictor)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        features = x
        mask = self.predictor(self.deconv_activation(self.deconv(features)))
        return mask, features


@RGB_MASK_HEAD_REGISTRY.register()
class GCNUpsampleHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super(GCNUpsampleHead, self).__init__()
        self.channels = cfg.MODEL.ROI_RGB_MASK_HEAD.IN_CHANNELS
        num_classes = cfg.DATASETS.NUM_CLASSES
        norm = cfg.MODEL.ROI_RGB_MASK_HEAD.NORM
        num_conv_layers = cfg.MODEL.ROI_RGB_MASK_HEAD.CONV_LAYERS
        num_gcn_layers = cfg.MODEL.ROI_RGB_MASK_HEAD.GCN_LAYERS
        num_upsample_layers = cfg.MODEL.ROI_RGB_MASK_HEAD.UPSAMPLE_LAYERS
        # CNN model
        self.cnn_layers = CNN(input_shape.channels, self.channels, norm, num_conv_layers)
        # GCN model
        self.gcn_layers = GCN(self.channels, num_gcn_layers)
        # Upsample model
        self.upsample_layers = Upsample(self.channels, num_classes, norm, num_upsample_layers)

    def forward(self, features):
        # If there are no detection, return random masks
        if features.shape[0] == 0:
            return torch.randn(0, 1, 2*features.shape[2], 2*features.shape[3], device=features.device), features
        # Apply CNN
        x = self.cnn_layers(features)
        # Create graph data for GCN
        adjacency_matrices = cosine_similarity_matrix(x)
        norm_adjacency_matrices = [normalize_adjacency_matrix(am) for am in adjacency_matrices]
        feature_maps = x.view(x.size(0), self.channels, -1).transpose(-2, -1)
        data_list = [create_graph_data(features, adj) for features, adj in zip(feature_maps, norm_adjacency_matrices)]
        # Apply GCN
        gcn_features = torch.stack([self.gcn_layers(d) for d in data_list], dim=0)
        gcn_features = gcn_features.view(x.size(0), self.channels, x.size(2), x.size(3))
        # Apply Upsample
        masks, roi_features = self.upsample_layers(gcn_features)
        return masks, roi_features


def build_rgb_mask_head(cfg, input_shape):
    name = cfg.MODEL.ROI_RGB_MASK_HEAD.NAME
    return RGB_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)