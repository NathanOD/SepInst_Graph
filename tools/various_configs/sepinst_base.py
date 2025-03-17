import torch
import torch.nn as nn
import wandb

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
#from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.structures import ImageList, Boxes, pairwise_iou

from custom_model.backbone import build_depth_backbone
from custom_model.criterion import build_matcher
from custom_model.proposal_generator import build_proposal_generator
from custom_model.fusion import build_fusion_module
from custom_model.roi_heads import build_rgb_mask_head, build_depth_mask_head, mask_loss
from .utils import nested_tensor_from_tensor_list


__all__ = ["SepInst"]


@META_ARCH_REGISTRY.register()
class SepInst(nn.Module):
    def __init__(self, cfg):
        super(SepInst, self).__init__()
        # Target device
        self.device = torch.device(cfg.MODEL.DEVICE)
        # Use custom backbone or not
        self.classic_backbone = cfg.MODEL.DEPTHBACKBONE.NAME.startswith("build_")
        # Use depth or not
        self.use_depth = cfg.MODEL.USE_DEPTH
        # Use double loss or not
        self.double_loss = cfg.MODEL.LOSS.DOUBLE_LOSS
        # Use dice loss or not
        self.dice_loss = cfg.MODEL.LOSS.DICE_LOSS
        # Backbone
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = {k: v.stride for k, v in backbone_shape.items()}
        self.in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
        # Proposal Generator
        self.proposal_generator = build_proposal_generator(cfg, backbone_shape)
        # RGB RoI Pooler
        self.rgb_mask_pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_RGB_MASK_HEAD.POOLER_RESOLUTION,
            scales=tuple(1.0 / self.feature_strides[k] for k in self.in_features),
            sampling_ratio=cfg.MODEL.ROI_RGB_MASK_HEAD.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_RGB_MASK_HEAD.POOLER_TYPE,
        )
        # RGB mask head
        self.rgb_mask_head = build_rgb_mask_head(
            cfg,
            ShapeSpec(channels=[backbone_shape[f] for f in self.in_features][0].channels,
                      width=cfg.MODEL.ROI_RGB_MASK_HEAD.POOLER_RESOLUTION,
                      height=cfg.MODEL.ROI_RGB_MASK_HEAD.POOLER_RESOLUTION)
        )
        if self.use_depth:
            # Depth backbone
            self.depthbackbone = build_depth_backbone(cfg)
            depthbackbone_shape = self.depthbackbone.output_shape()
            self.depth_feature_strides = {k: v.stride for k, v in depthbackbone_shape.items()}
            self.depth_in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
            # Depth RoI Pooler
            if self.classic_backbone:
                self.depth_mask_pooler = ROIPooler(
                    output_size=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_RESOLUTION,
                    scales=tuple(1.0 / self.depth_feature_strides[k] for k in self.depth_in_features),
                    sampling_ratio=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_SAMPLING_RATIO,
                    pooler_type=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_TYPE,
                )
            else:
                # if the depth backbone is built from scratch, we need to use the last stride
                last_stride = depthbackbone_shape[self.depth_in_features[-1]].stride
                self.depth_mask_pooler = ROIPooler(
                    output_size=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_RESOLUTION,
                    scales=[1.0 / last_stride],
                    sampling_ratio=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_SAMPLING_RATIO,
                    pooler_type=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_TYPE,
                )
            # Fusion module
            self.fusion_module = build_fusion_module(cfg)
            # RGBD mask head
            if self.classic_backbone:
                self.rgbd_mask_head = build_depth_mask_head(
                    cfg,
                    ShapeSpec(channels=[depthbackbone_shape[f] for f in self.depth_in_features][0].channels,
                              width=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_RESOLUTION,
                              height=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_RESOLUTION)
                )
            else:
                # if the depth backbone is built from scratch, we need to use the last stride
                self.rgbd_mask_head = build_depth_mask_head(
                    cfg,
                    ShapeSpec(channels=[depthbackbone_shape[f] for f in self.depth_in_features][-1].channels,
                              width=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_RESOLUTION,
                              height=cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_RESOLUTION)
                )
        # Matcher to assign box proposals to gt boxes
        #self.proposal_matcher = Matcher(
        #    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
        #    cfg.MODEL.ROI_HEADS.IOU_LABELS,
        #    allow_low_quality_matches=False,
        #)
        self.proposal_matcher = build_matcher(cfg)
        # Data and preprocessing
        self.num_classes = cfg.DATASETS.NUM_CLASSES
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # Inference
        self.mask_threshold = cfg.MODEL.MASK_THRESHOLD
        # Loss weights
        self.loss_weights = {
            'loss_fcos_loc': cfg.MODEL.LOSS.LOC_WEIGHT,
            'loss_fcos_ctr': cfg.MODEL.LOSS.CENTERNESS_WEIGHT,
            'loss_fcos_cls': cfg.MODEL.LOSS.CLASS_WEIGHT,
            'loss_mask_rgb': cfg.MODEL.LOSS.MASK_RGB_WEIGHT,
            'loss_mask_rgbd': cfg.MODEL.LOSS.MASK_RGBD_WEIGHT,
            'loss_dice_rgb': cfg.MODEL.LOSS.DICE_RGB_WEIGHT,
            'loss_dice_rgbd': cfg.MODEL.LOSS.DICE_RGBD_WEIGHT,
        }
        # Wandb config
        self.add_metrics = cfg.WANDB.ADD_METRICS
        if self.classic_backbone:
            depth_bb_name = f"R{cfg.MODEL.RESNETS.DEPTH}"
        else:
            depth_bb_name = cfg.MODEL.DEPTHBACKBONE.NAME
        if cfg.WANDB.ENABLED:
            self.wandb_run = wandb.init(
                project="SepInst",
                name = (
                    f"R{cfg.MODEL.RESNETS.DEPTH}-"
                    f"{depth_bb_name}-"
                    f"{cfg.MODEL.HEAD_NAME}-"
                    f"{cfg.MODEL.ROI_RGB_MASK_HEAD.POOLER_RESOLUTION}x{cfg.MODEL.ROI_RGBD_MASK_HEAD.POOLER_RESOLUTION}-"
                    f"{cfg.MODEL.FUSION.NAME}-"
                    f"{cfg.DATASETS.TRAIN[0][:-6]}"
                ),
                config={
                "batch_size": cfg.SOLVER.IMS_PER_BATCH,
                "base_learning_rate": cfg.SOLVER.BASE_LR,
                "steps": cfg.SOLVER.STEPS,
                "iterations": cfg.SOLVER.MAX_ITER,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                "checkpoint_period": cfg.SOLVER.CHECKPOINT_PERIOD,
                "optimizer": cfg.SOLVER.OPTIMIZER,
                "train_dataset": cfg.DATASETS.TRAIN,
                "val_dataset": cfg.DATASETS.TEST,
                "num_classes": cfg.DATASETS.NUM_CLASSES,
                })

    
    def normalizer(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        return image
    
    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, 32)
        depths = [x["depth"].to(self.device) for x in batched_inputs]
        depths = [self.normalizer(x) for x in depths]
        depths = ImageList.from_tensors(depths, 32)
        return images, depths


    def forward(self, batched_inputs):
        # Load images and depths
        images, depths = self.preprocess_inputs(batched_inputs)
        # Preprocess images and depths
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        if isinstance(depths, (list, torch.Tensor)):
            depths = nested_tensor_from_tensor_list(depths)
        # RGB backbone
        features = self.backbone(images.tensor)
        # Depth backbone
        if self.use_depth:
            features_depth = self.depthbackbone(depths.tensor)
        else:
            features_depth = None
        # Forward pass
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            return self.training_step(features, features_depth, images, gt_instances)
        else:
            return self.inference_step(features, features_depth, batched_inputs, images)

    def training_step(self, features, features_depth, images, gt_instances):
        # Proposal Generator
        rpn_proposals, rpn_losses = self.proposal_generator(images, features, gt_instances)
        # Modify proposals to fit the format of Detectron2 training Instances
        for rpn_proposal in rpn_proposals:
            rpn_proposal.proposal_boxes = rpn_proposal.pred_boxes
            rpn_proposal.objectness_logits = rpn_proposal.scores
            rpn_proposal._fields.pop('pred_boxes')
            rpn_proposal._fields.pop('scores')
            rpn_proposal._fields.pop('pred_classes')
            rpn_proposal._fields.pop('fpn_levels')
            rpn_proposal._fields.pop('locations')
        rpn_proposals = self.label_and_sample_proposals(rpn_proposals, gt_instances)
        del gt_instances
        # Mask Heads
        mask_losses, mask_metrics = self.forward_mask(features, features_depth, rpn_proposals)
        losses = {**rpn_losses, **mask_losses}
        weighted_losses = {k: v * self.loss_weights[k] for k, v in losses.items()}
        if hasattr(self, "wandb_run"):
            wandb.log(losses)
            if self.add_metrics:
                wandb.log(mask_metrics)
        return weighted_losses
    

    def inference_step(self, features, features_depth, batched_inputs, images):
        # Proposal Generator
        rpn_proposals = self.proposal_generator(images, features)
        # Mask Heads
        instances = self.forward_mask(features, features_depth, rpn_proposals)
        return self._postprocess(instances, batched_inputs, images.image_sizes, self.mask_threshold)
    
    def forward_mask(self, features, features_depth, instances):
        rgb_features_list = [features[f] for f in self.in_features]
        if self.use_depth:
            depth_features_list = [features_depth[f] for f in self.depth_in_features]
        del features
        if self.training:
            # The loss is only defined on positive proposals
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            # Poolers
            mask_features = self.rgb_mask_pooler(rgb_features_list, proposal_boxes)
            if self.use_depth:
                if self.classic_backbone:
                    depth_mask_features = self.depth_mask_pooler(depth_features_list, proposal_boxes)
                else:
                    # if the depth backbone is built from scratch, we need to use the last stride
                    depth_mask_features = self.depth_mask_pooler([depth_features_list[-1]], proposal_boxes)
            # RGB Mask Head
            rgb_masks, rgb_mask_features = self.rgb_mask_head(mask_features)
            # RGBD Head
            if self.use_depth:
                # Fusion Module
                fused_features = self.fusion_module(rgb_mask_features, depth_mask_features)
                # RGBD Mask Head
                rgbd_masks, _ = self.rgbd_mask_head(fused_features)
            # Compute losses
            if self.use_depth and self.double_loss:
                mask_losses_rgb, mask_metrics_rgb = mask_loss(rgb_masks, proposals, depth=False, use_dice_loss=self.dice_loss)
                mask_losses_rgbd, mask_metrics_rgbd = mask_loss(rgbd_masks, proposals, depth=True, use_dice_loss=self.dice_loss)
                mask_losses = {**mask_losses_rgb, **mask_losses_rgbd}
                mask_metrics = {**mask_metrics_rgb, **mask_metrics_rgbd}
            elif self.use_depth:
                mask_losses, mask_metrics = mask_loss(rgbd_masks, proposals, depth=True, use_dice_loss=self.dice_loss)
            else:
                mask_losses, mask_metrics = mask_loss(rgb_masks, proposals, depth=False, use_dice_loss=self.dice_loss)
            return mask_losses, mask_metrics
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            # Poolers
            mask_features = self.rgb_mask_pooler(rgb_features_list, pred_boxes)
            if self.use_depth:
                if self.classic_backbone:
                    depth_mask_features = self.depth_mask_pooler(depth_features_list, pred_boxes)
                else:
                    # if the depth backbone is built from scratch, we need to use the last stride
                    depth_mask_features = self.depth_mask_pooler([depth_features_list[-1]], pred_boxes)
            # RGB Mask Head
            rgb_masks, rgb_mask_features = self.rgb_mask_head(mask_features)
            # RGBD Head
            if self.use_depth:
                # Fusion Module
                fused_features = self.fusion_module(rgb_mask_features, depth_mask_features)
                # RGBD Mask Head
                rgbd_masks, _ = self.rgbd_mask_head(fused_features)
                # Inference
                mask_rcnn_inference(rgbd_masks, instances)
            else:
                # Inference
                mask_rcnn_inference(rgb_masks, instances)
            return instances


    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        # TODO : Regarder ça
        if True: #self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    
    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.nonzero((gt_classes != -1) & (gt_classes != self.num_classes)).squeeze(1)
        
        return sampled_idxs, gt_classes[sampled_idxs]
    

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes, mask_threshold):
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width, mask_threshold)
            processed_results.append({"instances": r})
        return processed_results
