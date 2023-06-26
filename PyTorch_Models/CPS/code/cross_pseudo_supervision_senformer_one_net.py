import math
import os
import statistics
import sys
import numpy
import wandb
from torch import nn
import torch.nn.functional as F

from PyTorch_Models.CPS.code.networks.senformer import SenFormer

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
from pathlib import Path
from PyTorch_Models.CEECNET.utils import Dotdict, detach_to_numpy, wb_mask

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from PyTorch_Models.CEECNET.dset import UnsupervisedSegmentationDataset, SegmentationDataset
from PyTorch_Models.CEECNET.eval import run_eval
from networks.unet import SmallUNet

from utils import losses, metrics

HPP_DEFAULT = Dotdict(dict(
    ### Senformer
    swin_tiny=True,
    branch_depth=4,
    ### CPS
    cps_weight=1,
    base_lr=0.00006,
    ### DATASET AND DATALOADER
    in_channels=1,
    nclasses=2,
    data_num="7_1",  # {7; 10; 15; 25; 50}_[[0,10]]
    e_name="101b",  # {101a; 101b; PPT}
    nb_crop_per_image=64,
    nuclei=True,  # if false, will be nucleoli
    batch_size=10,
    normalize=True,
    random_rescale=False,
    downsize=True,
    # If True, images are cropped to [dim_crop*dim_crop] then downsized to [dimensions_input*dimensions_input]
    # else images are directly cropped to [dimensions_input*dimensions_input]
    dim_crop=2048,
    dimensions_input=512,
    keep_empty_output_prob=0.001,
    ### MISC
    epochs=750,
    seed=42,
    loss_depth_init=0,
    train_data_path=None,
    train_masks_path=None,
    val_data_path=None,
    val_masks_path=None,
))

def lamdba_scaling(epoch):
    if epoch <= 5:
        return 1e-5
    elif epoch <= 20:
        m = (HPP_DEFAULT.cps_weight - 1e-5) / 15
        b = HPP_DEFAULT.cps_weight - 20 * m
        return m * epoch + b
    else:
        return HPP_DEFAULT.cps_weight


def add_weight_decay(net, l2_value=0.01, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name.split(".")[-1] in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]



def poly_lr_linear_warmup(base_lr, iter, max_iter):
    if iter < 1500:
        slope = base_lr / 1500
        lr = slope * iter
    else:
        lr = base_lr * (1 - iter / max_iter)
    return lr


if __name__ == "__main__":
    world_size = 1
    global_rank = 0
    nuclei_string = "nuclei" if HPP_DEFAULT.nuclei else "nucleoli"
    run = wandb.init(project="DTC", config=HPP_DEFAULT, settings=wandb.Settings(start_method="fork"),
                     tags=["SSL", "CutMix", "batch={}".format(world_size * HPP_DEFAULT.batch_size), "n_gpu={}".format(world_size),
                           "downsize={}".format(HPP_DEFAULT.downsize), nuclei_string])
    model_dir = os.path.join("./models", run.id)
    fake_batch_size, diff = (HPP_DEFAULT.batch_size // 2 + 1, 1) if HPP_DEFAULT.batch_size % 2 != 0 else (HPP_DEFAULT.batch_size // 2, 0)
    Path(model_dir).mkdir(parents=True, exist_ok=False)
    RUN_ID = run.id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("init successful")
    base_lr = HPP_DEFAULT.base_lr
    model = SenFormer(num_heads=8, branch_depth=HPP_DEFAULT.branch_depth, in_chans=1, num_classes=HPP_DEFAULT.nclasses,
                      mlp_ratio=4.,
                      qkv_bias=True,
                      qk_scale=None, drop=0.,
                      attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1.e-15,
                      align_corners=False, swin_tiny=HPP_DEFAULT.swin_tiny).train().to(device)

    d_train = UnsupervisedSegmentationDataset(HPP_DEFAULT, HPP_DEFAULT.train_data_path, HPP_DEFAULT.train_masks_path,
                                              train=True)
    d_val = SegmentationDataset(HPP_DEFAULT, HPP_DEFAULT.val_data_path, HPP_DEFAULT.val_masks_path, train=False)
    train_loader = DataLoader(d_train, fake_batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(d_val, HPP_DEFAULT.batch_size, pin_memory=True)
    params1 = add_weight_decay(model,
                               skip_list=["absolute_pos_embed", "relative_position_bias_table", "norm", "queries"])
    optimizer = optim.Adam(model.parameters(), lr=HPP_DEFAULT.base_lr)
    lr_ = base_lr
    best_loss = math.inf
    best_dice = 0
    iter = 0
    eps = 1.e-15
    max_iter = len(train_loader) * HPP_DEFAULT.epochs
    for epoch_num in range(HPP_DEFAULT.epochs):
        to_log = {}
        losses_val, dices_val = [], []
        losses_train, cps_, sup_  = [], [], []
        # l_images for labeled images, ul_images for unlabeled images
        for i_batch, (l_images, targets, ul_images) in enumerate(train_loader):
            optimizer.zero_grad()
            if diff > 0: # remove last ul to get to odd batch size if needed
                ul_images = ul_images[:-1]
            ul_batch, l_batch, targets = ul_images.to(device), l_images.to(device), targets[0].to(device)
            outsup, outunsup = model(l_batch), model(ul_batch)

            # CPS LOSS
            # generate Y from ensemble
            outs_ensemble = torch.cat([outsup[1], outunsup[1]], dim=0)
            soft_pred = F.interpolate(torch.log(torch.clamp(outs_ensemble, min=eps)), size=targets.shape[2:], mode="bilinear")
            max_ensemble = torch.argmax(soft_pred, dim=1).unsqueeze(1)
            max_ensemble = max_ensemble.long()

            # Supervise learners with ensemble pred
            logits_sup = [F.interpolate(input=outsup[0][i], size=targets.shape[2:], mode='bilinear',
                                          align_corners=False) for i in range(len(outsup[0]))]
            logits_unsup = [F.interpolate(input=outunsup[0][i], size=targets.shape[2:], mode='bilinear',
                                            align_corners=False) for i in range(len(outunsup[0]))]
            logits_l = [torch.cat((logits_sup[i], logits_unsup[i]), dim=0) for i in range(len(logits_sup))]
            cps_logits = [F.cross_entropy(logits_l[i], max_ensemble) for i in range(len(logits_l))]
            cps = torch.stack(cps_logits, dim=0).mean()
            cps = cps * lamdba_scaling(epoch_num)

            # Supervised loss
            sup_loss = model.losses(outsup, targets.long())

            loss = cps + sup_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            cps_.append(cps.item())
            losses_train.append(loss.item())
            sup_.append(sup_loss.item())


            # update lr
            lr_ = poly_lr_linear_warmup(base_lr, iter, max_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter += 1

        model.eval()
        with torch.no_grad():
            for i_batch, (images, targets) in enumerate(val_loader):
                images, targets = images.to(device), targets[0].to(device)
                outputs = model(images)
                pred = F.interpolate(outputs[1], size=targets.shape[2:], mode="bilinear")
                pred = torch.argmax(pred, dim=1)
                dc = metrics.dice(pred, targets)
                dices_val.append(dc.item())
        model.train()

        dice_val = statistics.mean(dices_val)

        # log
        to_log["lr"] = lr_
        to_log["lambda"] = lamdba_scaling(epoch_num)
        to_log["train/loss"] = statistics.mean(losses_train)
        to_log["train/cps"] = statistics.mean(cps_)
        to_log["train/sup"] = statistics.mean(sup_)
        to_log["val/dice"] = dice_val

        # save model
        if dice_val > best_dice:
            best_dice = dice_val
            to_log["best_dice"] = best_dice
            # save model
            torch.save({
                "epoch": epoch_num,
                "state_dict": model.state_dict(),
            }, os.path.join(model_dir, "model_best_dice.pt"))
        wandb.log(to_log)
    os.chdir("../../CEECNET")
    wandb.finish()
    run_eval(RUN_ID)
