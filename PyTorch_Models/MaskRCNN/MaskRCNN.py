import torch
import torchvision

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import misc as misc_nn_ops

from collections import OrderedDict

import torch.nn.functional as F

# mes couilles en string!!!

# RPN parameters
rpn_pre_nms_top_n_train = 2048  # number of proposals to keep before applying NMS during training
rpn_pre_nms_top_n_test = 1024  # number of proposals to keep before applying NMS during testing
rpn_post_nms_top_n_train = 2048  # number of proposals to keep after applying NMS during training
rpn_post_nms_top_n_test = 1024  # number of proposals to keep after applying NMS during testing
rpn_nms_thresh = 0.7  # NMS threshold used for postprocessing the RPN proposals
rpn_fg_iou_thresh = 0.7  # minimum IoU between the anchor and the GT box so that they can be considered as positive during training of the RPN.
rpn_bg_iou_thresh = 0.3  # maximum IoU between the anchor and the GT box so that they can be considered as negative during training of the RPN.
rpn_batch_size_per_image = 256  # number of anchors that are sampled during training of the RPN for computing the loss
rpn_positive_fraction = 0.5  # proportion of positive anchors in a mini-batch during training of the RPN

# Box parameters
box_score_thresh = 0.05  # during inference, only return proposals with a classification score greater than box_score_thresh
box_nms_thresh = 0.5  # NMS threshold for the prediction head. Used during inference
box_detections_per_img = 101  # maximum number of detections per image, for all classes.
box_fg_iou_thresh = 0.5  # minimum IoU between the proposals and the GT box so that they can be considered as positive during training of the classification head
box_bg_iou_thresh = 0.5  # maximum IoU between the proposals and the GT box so that they can be considered as negative during training of the classification head
box_batch_size_per_image = 512  # number of proposals that are sampled during training of the classification head
box_positive_fraction = 0.25  # proportion of positive proposals in a mini-batch during training of the classification head

from torchvision.models.detection import roi_heads


def ResNet50(nbClasses: int, HiddenLayer: int = 512, Pretrained: bool = True, PretrainedBackbone: bool = True,
			TrainableBackboneLayers: int = 3,
			Anchors: str = None, AddLayersToPredictor=0, Loss=None):
	if Loss is not None:
		roi_heads.maskrcnn_loss = Loss
	
	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=Pretrained,
																pretrained_backbone=PretrainedBackbone,
																# trainable_backbone_layers=TrainableBackboneLayers,
																rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
																rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
																rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
																rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
																rpn_nms_thresh=rpn_nms_thresh,
																rpn_fg_iou_thresh=rpn_fg_iou_thresh,
																rpn_bg_iou_thresh=rpn_bg_iou_thresh,
																rpn_batch_size_per_image=rpn_batch_size_per_image,
																rpn_positive_fraction=rpn_positive_fraction,
																box_score_thresh=box_score_thresh,
																box_nms_thresh=box_nms_thresh,
																box_detections_per_img=box_detections_per_img,
																box_fg_iou_thresh=box_fg_iou_thresh,
																box_bg_iou_thresh=box_bg_iou_thresh,
																box_batch_size_per_image=box_batch_size_per_image,
																box_positive_fraction=box_positive_fraction)
	
	if Anchors == "Nuclei":
		anchor_generator = AnchorGenerator(sizes=tuple([(16, 32, 64) for _ in range(5)]),
										   aspect_ratios=tuple([(0.5, 1.0, 1.5) for _ in range(5)]))
		model.rpn.anchor_generator = anchor_generator
		model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
	
	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nbClasses)
	
	# now get the number of input features for the mask classifier
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	
	if 0 < AddLayersToPredictor:  # and replace the mask predictor with a new one
		model.roi_heads.mask_predictor = MaskRCNNPredictorTuned(in_features_mask, HiddenLayer, nbClasses,
																AddLayersToPredictor)
	else:
		model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, HiddenLayer, nbClasses)
	
	return model


def ResNet(Version: int = 50, nbClasses: int = 2, HiddenLayer: int = 512, PretrainedBackbone: bool = True,
			TrainableBackboneLayers: int = 3, Anchors: str = None,
			norm_layer=misc_nn_ops.FrozenBatchNorm2d, AddLayersToPredictor=0, Loss=None):
	Versions = {18, 34, 50, 101, 152}
	if Version not in Versions:
		raise Exception("ResNet version must be among " + str(Versions) + ".")
	
	if Version == 18:
		backbone = torchvision.models.resnet18(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	elif Version == 34:
		backbone = torchvision.models.resnet34(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	elif Version == 50:
		backbone = torchvision.models.resnet50(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	elif Version == 101:
		backbone = torchvision.models.resnet101(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	elif Version == 152:
		backbone = torchvision.models.resnet152(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	else:
		Exception("Must not occur.")
	
	return _ResNet(backbone, nbClasses, HiddenLayer, TrainableBackboneLayers, Anchors, AddLayersToPredictor, Loss)


def ResNext(Version: int = 50, nbClasses: int = 2, HiddenLayer: int = 512, PretrainedBackbone: bool = True,
			TrainableBackboneLayers: int = 3, Anchors: str = None,
			norm_layer=misc_nn_ops.FrozenBatchNorm2d, AddLayersToPredictor=0, Loss=None):
	Versions = {50, 101}
	if Version not in Versions:
		raise Exception("ResNet version must be among " + str(Versions) + ".")
	
	if Version == 50:
		backbone = torchvision.models.resnext50_32x4d(pretrained=PretrainedBackbone, norm_layer=norm_layer,
													progress=True)
	elif Version == 101:
		backbone = torchvision.models.resnext101_32x8d(pretrained=PretrainedBackbone, norm_layer=norm_layer,
													progress=True)
	else:
		Exception("Must not occur.")
	
	return _ResNet(backbone, nbClasses, HiddenLayer, TrainableBackboneLayers, Anchors, AddLayersToPredictor, Loss)


def _ResNet(backbone, nbClasses: int = 2, HiddenLayer: int = 512, TrainableBackboneLayers: int = 3, Anchors: str = None,
			AddLayersToPredictor=0, Loss=None):
	if Loss is not None:
		roi_heads.maskrcnn_loss = Loss
	
	if TrainableBackboneLayers < 0 or 5 < TrainableBackboneLayers:  # select layers that wont be frozen
		raise Exception("TrainableBackboneLayers must be within range [0,5].")
	
	layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:TrainableBackboneLayers]
	
	for name, parameter in backbone.named_parameters():  # freeze layers only if pretrained backbone is used
		if all([not name.startswith(layer) for layer in layers_to_train]):
			parameter.requires_grad_(False)
	
	return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
	
	in_channels_stage2 = backbone.inplanes // 8
	in_channels_list = [
		in_channels_stage2,
		in_channels_stage2 * 2,
		in_channels_stage2 * 4,
		in_channels_stage2 * 8,
	]
	out_channels = 256
	
	fpnbackbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
	
	model = MaskRCNN(fpnbackbone, 91,
					rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
					rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
					rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
					rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
					rpn_nms_thresh=rpn_nms_thresh,
					rpn_fg_iou_thresh=rpn_fg_iou_thresh,
					rpn_bg_iou_thresh=rpn_bg_iou_thresh,
					rpn_batch_size_per_image=rpn_batch_size_per_image,
					rpn_positive_fraction=rpn_positive_fraction,
					box_score_thresh=box_score_thresh,
					box_nms_thresh=box_nms_thresh,
					box_detections_per_img=box_detections_per_img,
					box_fg_iou_thresh=box_fg_iou_thresh,
					box_bg_iou_thresh=box_bg_iou_thresh,
					box_batch_size_per_image=box_batch_size_per_image,
					box_positive_fraction=box_positive_fraction)
	
	if Anchors == "Nuclei":
		anchor_generator = AnchorGenerator(sizes=tuple([(16, 32, 64) for _ in range(5)]),
											aspect_ratios=tuple([(0.5, 1.0, 1.5) for _ in range(5)]))
		model.rpn.anchor_generator = anchor_generator
		model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
	
	# Get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# Replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nbClasses)
	
	# Get the number of input features for the mask classifier
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	
	if 0 < AddLayersToPredictor:  # and replace the mask predictor with a new one
		model.roi_heads.mask_predictor = MaskRCNNPredictorTuned(in_features_mask, HiddenLayer, nbClasses,
																AddLayersToPredictor)
	else:
		model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, HiddenLayer, nbClasses)
	
	# if Loss is not None: model.roi_heads.mask_rcnn_loss = Loss
	
	return model


def Trainable(model, part: str, value: bool):
	Parts = {'backbone', 'rpn', 'roi'}
	if part not in Parts:
		raise Exception("Unknown part: '" + part + "'. Must be choosen among " + str(Parts))
	
	for name, param in model.named_parameters():
		if name.startswith(part):
			param.requires_grad_(value)


"""
class MaskRCNNHeads(nn.Sequential):
	def __init__(self, in_channels, layers=(256, 256, 256, 256), dilation=1):
		 Arguments:
				in_channels (int): number of input channels
				layers (list): feature dimensions of each FCN layer
				dilation (int): dilation rate of kernel
		
		d = OrderedDict()
		next_feature = in_channels
		for layer_idx, layer_features in enumerate(layers, 1):
			d["mask_fcn{}".format(layer_idx)] = misc_nn_ops.Conv2d(next_feature, layer_features, kernel_size=3, stride=1, padding=dilation,
																	dilation=dilation)
			d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
			next_feature = layer_features

		super(MaskRCNNHeads, self).__init__(d)
		for name, param in self.named_parameters():
			if "weight" in name:
				nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
			# elif "bias" in name:
			#	 nn.init.constant_(param, 0)
"""


class MaskRCNNPredictorTuned(torch.nn.Sequential):
	def __init__(self, in_channels, dim_reduced, num_classes, toAdd):
		if toAdd == 1:
			super(MaskRCNNPredictorTuned, self).__init__(OrderedDict([
				("conv5_mask", torch.nn.Conv2d(in_channels, in_channels, 3)),
				("relu", torch.nn.ReLU(inplace=True)),
				("conv6_mask", torch.nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
				("relu", torch.nn.ReLU(inplace=True)),
				("mask_fcn_logits", misc_nn_ops.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
			]))
		elif toAdd == 2:
			super(MaskRCNNPredictorTuned, self).__init__(OrderedDict([
				("conv5_mask", torch.nn.Conv2d(in_channels, in_channels, 3)),
				("relu", torch.nn.ReLU(inplace=True)),
				("conv6_mask", torch.nn.Conv2d(in_channels, in_channels, 3)),
				("relu", torch.nn.ReLU(inplace=True)),
				("conv7_mask", torch.nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
				("relu", torch.nn.ReLU(inplace=True)),
				("mask_fcn_logits", torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
			]))
		else:
			raise Exception("Unsupported value.")
		
		for name, param in self.named_parameters():
			if "weight" in name:
				torch.nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
	# elif "bias" in name:
	#	torch.nn.init.constant_(param, 0)


def Tanimoto(Input, Target, Axis=(1, 2, 3), Smooth=1.0e-5):
	intersection = (Input * Target).sum(dim=Axis)
	cIn = torch.sum(Input ** 2, dim=Axis)
	cTar = torch.sum(Target ** 2, dim=Axis)
	return (intersection + Smooth) / (cIn + cTar - intersection + Smooth)


def Tanimoto_Dual(Input, Target, Axis=(1, 2, 3), Smooth=1.0e-5):
	Foreground = Tanimoto(Input, Target, Axis=Axis, Smooth=Smooth)
	Background = Tanimoto(1. - Input, 1. - Target, Axis=Axis, Smooth=Smooth)
	return 0.5 * (Foreground + Background)


def DiceLoss_Tanimoto(Input, Target, Axis=(1, 2, 3), Smooth=1.0e-5):
	return 1.0 - Tanimoto(Input, Target, Axis=Axis, Smooth=Smooth).mean()


def DiceLoss_Tanimoto_Dual(Input, Target, Axis=(1, 2, 3), Smooth=1.0e-5):
	return 1.0 - Tanimoto_Dual(Input, Target, Axis=Axis, Smooth=Smooth).mean()


def Loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
	# type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor])
	""" Arguments:
			proposals (list[BoxList])
			mask_logits (Tensor)
			targets (list[BoxList])
		Return:
			mask_loss (Tensor): scalar tensor containing the loss
	"""
	
	discretization_size = mask_logits.shape[-1]
	labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
	mask_targets = [roi_heads.project_masks_on_boxes(m, p, i, discretization_size)
					for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)]
	
	labels = torch.cat(labels, dim=0)
	mask_targets = torch.cat(mask_targets, dim=0)
	
	# torch.mean (in binary_cross_entropy_with_logits) doesn't accept empty tensors, so handle it separately
	if mask_targets.numel() == 0:
		return mask_logits.sum() * 0
	
	tmp = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
	# print(mask_targets.shape) # torch.Size([155, 28, 28])
	# print(tmp.shape) # torch.Size([155, 28, 28])
	# mask_loss = F.binary_cross_entropy_with_logits(tmp, mask_targets) # Default.
	# mask_loss = F.smooth_l1_loss(tmp, mask_targets)
	# mask_loss = F.l1_loss(tmp, mask_targets)
	# mask_loss = F.mse_loss(tmp, mask_targets)
	# mask_loss = F.margin_ranking_loss(tmp, mask_targets)
	# mask_loss = F.hinge_embedding_loss(tmp, mask_targets)
	# mask_loss = F.soft_margin_loss(tmp, mask_targets)
	# mask_loss = F.multi_margin_loss(tmp, mask_targets)
	mask_loss = DiceLoss_Tanimoto_Dual(tmp, mask_targets, Axis=(0, 1, 2))
	
	return mask_loss


def ResNetOld(Version: int = 50, nbClasses: int = 2, HiddenLayer: int = 512, PretrainedBackbone: bool = True,
			TrainableBackboneLayers: int = 5, Anchors: str = None,
			norm_layer=misc_nn_ops.FrozenBatchNorm2d):
	Versions = {18, 34, 50, 101, 152}
	if Version not in Versions:
		raise Exception("ResNet version must be among " + str(Versions) + ".")
	
	if TrainableBackboneLayers < 0 or 5 < TrainableBackboneLayers:  # select layers that wont be frozen
		raise Exception("TrainableBackboneLayers must be within range [0,5].")
	layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:TrainableBackboneLayers]
	
	if Version == 18:
		backbone = torchvision.models.resnet18(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	elif Version == 34:
		backbone = torchvision.models.resnet34(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	elif Version == 50:
		backbone = torchvision.models.resnet50(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	elif Version == 101:
		backbone = torchvision.models.resnet101(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	elif Version == 152:
		backbone = torchvision.models.resnet152(pretrained=PretrainedBackbone, norm_layer=norm_layer, progress=True)
	else:
		Exception("Must not occur.")
	
	for name, parameter in backbone.named_parameters():  # freeze layers only if pretrained backbone is used
		if all([not name.startswith(layer) for layer in layers_to_train]):
			parameter.requires_grad_(False)
	
	return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
	
	in_channels_stage2 = backbone.inplanes // 8
	in_channels_list = [
		in_channels_stage2,
		in_channels_stage2 * 2,
		in_channels_stage2 * 4,
		in_channels_stage2 * 8,
	]
	out_channels = 256
	
	fpnbackbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
	
	model = MaskRCNN(fpnbackbone, 91,
					 rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
					 rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
					 rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
					 rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
					 rpn_nms_thresh=rpn_nms_thresh,
					 rpn_fg_iou_thresh=rpn_fg_iou_thresh,
					 rpn_bg_iou_thresh=rpn_bg_iou_thresh,
					 rpn_batch_size_per_image=rpn_batch_size_per_image,
					 rpn_positive_fraction=rpn_positive_fraction,
					 box_score_thresh=box_score_thresh,
					 box_nms_thresh=box_nms_thresh,
					 box_detections_per_img=box_detections_per_img,
					 box_fg_iou_thresh=box_fg_iou_thresh,
					 box_bg_iou_thresh=box_bg_iou_thresh,
					 box_batch_size_per_image=box_batch_size_per_image,
					 box_positive_fraction=box_positive_fraction)
	
	if Anchors == "Nuclei":
		anchor_generator = AnchorGenerator(sizes=tuple([(16, 32, 64) for _ in range(5)]),
										   aspect_ratios=tuple([(0.5, 1.0, 1.5) for _ in range(5)]))
		model.rpn.anchor_generator = anchor_generator
		model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
	
	# Get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# Replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nbClasses)
	
	# Get the number of input features for the mask classifier
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	
	# Replace the mask predictor with a new one
	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, HiddenLayer, nbClasses)
	
	return model


def ResNextOld(Version: int = 50, nbClasses: int = 2, HiddenLayer: int = 512, PretrainedBackbone: bool = True,
			   TrainableBackboneLayers: int = 5, Anchors: str = None,
			   norm_layer=misc_nn_ops.FrozenBatchNorm2d):
	Versions = {50, 101}
	if Version not in Versions:
		raise Exception("ResNext version must be among " + str(Versions) + ".")
	
	if TrainableBackboneLayers < 0 or 5 < TrainableBackboneLayers:  # select layers that wont be frozen
		raise Exception("TrainableBackboneLayers must be within range [0,5].")
	layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:TrainableBackboneLayers]
	
	if Version == 50:
		backbone = torchvision.models.resnext50_32x4d(pretrained=PretrainedBackbone, norm_layer=norm_layer,
													  progress=True)
	elif Version == 101:
		backbone = torchvision.models.resnext101_32x8d(pretrained=PretrainedBackbone, norm_layer=norm_layer,
													   progress=True)
	else:
		Exception("Must not occur.")
	
	for name, parameter in backbone.named_parameters():  # freeze layers only if pretrained backbone is used
		if all([not name.startswith(layer) for layer in layers_to_train]):
			parameter.requires_grad_(False)
	
	return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
	
	in_channels_stage2 = backbone.inplanes // 8
	in_channels_list = [
		in_channels_stage2,
		in_channels_stage2 * 2,
		in_channels_stage2 * 4,
		in_channels_stage2 * 8,
	]
	out_channels = 256
	
	fpnbackbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
	
	model = MaskRCNN(fpnbackbone, 91,
					 rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
					 rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
					 rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
					 rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
					 rpn_nms_thresh=rpn_nms_thresh,
					 rpn_fg_iou_thresh=rpn_fg_iou_thresh,
					 rpn_bg_iou_thresh=rpn_bg_iou_thresh,
					 rpn_batch_size_per_image=rpn_batch_size_per_image,
					 rpn_positive_fraction=rpn_positive_fraction,
					 box_score_thresh=box_score_thresh,
					 box_nms_thresh=box_nms_thresh,
					 box_detections_per_img=box_detections_per_img,
					 box_fg_iou_thresh=box_fg_iou_thresh,
					 box_bg_iou_thresh=box_bg_iou_thresh,
					 box_batch_size_per_image=box_batch_size_per_image,
					 box_positive_fraction=box_positive_fraction)
	
	if Anchors == "Nuclei":
		anchor_generator = AnchorGenerator(sizes=tuple([(16, 32, 64) for _ in range(5)]),
										   aspect_ratios=tuple([(0.5, 1.0, 1.5) for _ in range(5)]))
		model.rpn.anchor_generator = anchor_generator
		model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
	
	# Get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# Replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nbClasses)
	
	# Get the number of input features for the mask classifier
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	
	# Replace the mask predictor with a new one
	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, HiddenLayer, nbClasses)
	
	return model
