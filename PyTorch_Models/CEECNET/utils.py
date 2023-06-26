import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import wandb


def get_norm(name, channels, norm_groups=None):
    if name == 'BatchNorm':
        return nn.BatchNorm2d(channels)
    elif name == 'InstanceNorm':
        return nn.InstanceNorm2d(channels)
    # elif (name == 'LayerNorm'):
    #     return nn.LayerNorm(axis=axis)
    elif name == 'GroupNorm' and norm_groups is not None:
        return nn.GroupNorm(num_groups=norm_groups, num_channels=channels)  # applied to channel axis
    else:
        raise NotImplementedError


def detach_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().clone().cpu().numpy()


def labels(classes):
    l = {}
    for i, label in enumerate(classes):
        l[i] = label
    return l


def wb_mask(bg_img, pred_mask, true_mask, classes_out, classes_gt=None):
    if classes_gt is None:
        classes_gt = classes_out
    return wandb.Image(bg_img, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": labels(classes_out)},
        "ground truth": {"mask_data": true_mask, "class_labels": labels(classes_gt)}})


def add_tags(run, tags):
    """add tags to wandb run"""
    api = wandb.Api()
    run_ = api.run("{}/{}".format(run.project, run.id))
    for tag in tags:
        run_.tags.append(tag)
    run_.update()


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dice_coeff(y_pred, y_true):
    smooth = 1.

    iflat = y_pred.view(-1)
    tflat = y_true.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.pow(2).sum() + tflat.pow(2).sum() + smooth)


def dice_coef_loss(y_pred, y_true):
    return 1. - dice_coeff(y_pred, y_true)


def get_boundary(labels, _kernel_size=(3, 3)):
    label = detach_to_numpy(labels)
    for channel in range(label.shape[0]):
        temp = cv2.Canny(label[channel], 0, 1)
        label[channel] = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_CROSS, _kernel_size), iterations=1)

    label = label.astype(np.float32)
    label /= 255.
    label = label.astype(np.uint8)
    return label


def get_distance(labels):
    label = detach_to_numpy(labels)
    dists = np.empty_like(label, dtype=np.float32)
    for channel in range(label.shape[0]):
        dist = cv2.distanceTransform(label[channel], cv2.DIST_L2, 0)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dists[channel] = dist

    return dists


def get_smp_model(config):
    if config.model_smp == "Unet":
        model = smp.Unet(
            encoder_depth=config.encoder_depth,
            decoder_channels=config.decoder_channels,
            encoder_name=config.encoder_smp,
            encoder_weights=None,
            activation=config.activation_smp,
            in_channels=config.in_channels,
            classes=config.nclasses
        )
    elif config.model_smp == "Unet++":
        model = smp.UnetPlusPlus(
            encoder_name=config.encoder_smp,
            encoder_depth=config.encoder_depth,
            encoder_weights=None,
            activation=config.activation_smp,
            decoder_channels=config.decoder_channels,
            in_channels=config.in_channels,
            classes=config.nclasses
        )
    elif config.model_smp == "Linknet":
        model = smp.Linknet(
            encoder_name=config.encoder_smp,
            encoder_weights=None,
            activation=config.activation_smp,
            in_channels=config.in_channels,
            classes=config.nclasses
        )
    elif config.model_smp == "FPN":
        model = smp.FPN(
            encoder_name=config.encoder_smp,
            encoder_weights=None,
            activation=config.activation_smp,
            in_channels=config.in_channels,
            classes=config.nclasses
        )
    elif config.model_smp == "DeepLabV3+":
        model = smp.DeepLabV3Plus(
            encoder_name=config.encoder_smp,
            encoder_weights=None,
            activation=config.activation_smp,
            in_channels=config.in_channels,
            classes=config.nclasses
        )
    else:
        raise Exception("Model not found in smp library, check that config.model_smp is one of {Unet; Unet++; Linknet; FPN; DeepLabV3+}")
    return model


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2