import torch
from torch import nn
import fvcore.nn.weight_init as weight_init

from detectron2.layers import Conv2d, get_norm
from detectron2.utils.registry import Registry


FUSION_REGISTRY = Registry("Fusion Module")
FUSION_REGISTRY.__doc__ = "Registry for fusion modules"


@FUSION_REGISTRY.register()
class ConvFusion(nn.Module):
    def __init__(self, cfg):
        super(ConvFusion, self).__init__()
        rgb_channels = cfg.MODEL.FUSION.IN_CHANNELS
        #depth_channels = cfg.MODEL.DEPTHBACKBONE.CONV_CHANNELS[-1]
        self.conv = Conv2d(2*rgb_channels,
                           #rgb_channels+depth_channels,
                           rgb_channels,
                           kernel_size=1,
                           norm=get_norm(cfg.MODEL.FUSION.NORM, rgb_channels))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        weight_init.c2_msra_fill(self.conv)

    def forward(self, rgb_features, depth_features):
        x = torch.cat((rgb_features, depth_features), dim=1)
        fused_features = self.conv(x)
        fused_features = self.activation(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features
    

@FUSION_REGISTRY.register()
class HadamardFusion(nn.Module):
    def __init__(self, cfg):
        super(HadamardFusion, self).__init__()
        self.rgb_channels = cfg.MODEL.FUSION.IN_CHANNELS
        self.depth_channels = cfg.MODEL.FUSION.IN_CHANNELS
        #self.depth_channels = cfg.MODEL.DEPTHBACKBONE.CONV_CHANNELS[-1]

        self.conv = Conv2d(self.rgb_channels,
                           self.rgb_channels,
                           kernel_size=1,
                           norm=get_norm(cfg.MODEL.FUSION.NORM, self.rgb_channels))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self._initialize_weights()

        assert self.rgb_channels == self.depth_channels, "The number of channels must match for multiplication and addition"

    def _initialize_weights(self):
        weight_init.c2_msra_fill(self.conv)

    def forward(self, rgb_features, depth_features):
        assert rgb_features.size(1) == self.rgb_channels, "Channel mismatch for RGB features"
        assert depth_features.size(1) == self.depth_channels, "Channel mismatch for depth features"

        fused_features = rgb_features * depth_features
        fused_features = self.conv(fused_features)
        fused_features = self.activation(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features


@FUSION_REGISTRY.register()
class MultAddFusion(nn.Module):
    def __init__(self, cfg):
        super(MultAddFusion, self).__init__()
        self.rgb_channels = cfg.MODEL.FUSION.IN_CHANNELS
        self.depth_channels = cfg.MODEL.FUSION.IN_CHANNELS
        #self.depth_channels = cfg.MODEL.DEPTHBACKBONE.CONV_CHANNELS[-1]

        self.conv = Conv2d(self.rgb_channels,
                           self.rgb_channels,
                           kernel_size=1,
                           norm=get_norm(cfg.MODEL.FUSION.NORM, self.rgb_channels))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self._initialize_weights()

        assert self.rgb_channels == self.depth_channels, "The number of channels must match for multiplication and addition"

    def _initialize_weights(self):
        weight_init.c2_msra_fill(self.conv)

    def forward(self, rgb_features, depth_features):
        assert rgb_features.size(1) == self.rgb_channels, "Channel mismatch for RGB features"
        assert depth_features.size(1) == self.depth_channels, "Channel mismatch for depth features"

        fused_features = rgb_features * depth_features + rgb_features
        fused_features = self.conv(fused_features)
        fused_features = self.activation(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features


@FUSION_REGISTRY.register()
class WeightedSumFusion(nn.Module):
    def __init__(self, cfg):
        super(WeightedSumFusion, self).__init__()
        self.rgb_channels = cfg.MODEL.FUSION.IN_CHANNELS
        self.depth_channels = cfg.MODEL.FUSION.IN_CHANNELS
        #self.depth_channels = cfg.MODEL.DEPTHBACKBONE.CONV_CHANNELS[-1]

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.conv = Conv2d(self.rgb_channels,
                           self.rgb_channels,
                           kernel_size=1,
                           norm=get_norm(cfg.MODEL.FUSION.NORM, self.rgb_channels))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self._initialize_weights()

        assert self.rgb_channels == self.depth_channels, "The number of channels must match for multiplication and addition"

    def _initialize_weights(self):
        weight_init.c2_msra_fill(self.conv)

    def forward(self, rgb_features, depth_features):
        assert rgb_features.size(1) == self.rgb_channels, "Channel mismatch for RGB features"
        assert depth_features.size(1) == self.depth_channels, "Channel mismatch for depth features"

        fused_features = self.alpha * rgb_features + self.beta * depth_features
        fused_features = self.activation(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features
    

@FUSION_REGISTRY.register()
class AttentionFusion(nn.Module):
    def __init__(self, cfg):
        super(AttentionFusion, self).__init__()
        self.rgb_channels = cfg.MODEL.FUSION.IN_CHANNELS
        self.depth_channels = cfg.MODEL.FUSION.IN_CHANNELS
        #self.depth_channels = cfg.MODEL.DEPTHBACKBONE.CONV_CHANNELS[-1]

        self.attention = nn.Sequential(
            Conv2d(self.rgb_channels + self.depth_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv = Conv2d(self.rgb_channels + self.depth_channels,
                           self.rgb_channels,
                           kernel_size=1,
                           norm=get_norm(cfg.MODEL.FUSION.NORM, self.rgb_channels))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self._initialize_weights()

        assert self.rgb_channels == self.depth_channels, "The number of channels must match for multiplication and addition"

    def _initialize_weights(self):
        weight_init.c2_msra_fill(self.conv)

    def forward(self, rgb_features, depth_features):
        assert rgb_features.size(1) == self.rgb_channels, "Channel mismatch for RGB features"
        assert depth_features.size(1) == self.depth_channels, "Channel mismatch for depth features"
        
        concatenated = torch.cat((rgb_features, depth_features), dim=1)
        attention_weights = self.attention(concatenated)
        fused_features = rgb_features * attention_weights + depth_features * (1 - attention_weights)
        fused_features = self.conv(torch.cat((fused_features, depth_features), dim=1))
        fused_features = self.activation(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features


def build_fusion_module(cfg):
    name = cfg.MODEL.FUSION.NAME
    return FUSION_REGISTRY.get(name)(cfg)