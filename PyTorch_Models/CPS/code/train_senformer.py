import math
import os
import random
import statistics
import sys
import numpy as np
import wandb

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
from pathlib import Path
from PyTorch_Models.CEECNET.utils import Dotdict
from PyTorch_Models.CEECNET.eval import run_eval
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PyTorch_Models.CEECNET.dset import SegmentationDataset
from networks.senformer import SenFormer

from utils import ramps, losses, metrics

HPP_DEFAULT = Dotdict(dict(
    ### DATASET AND DATALOADER
    in_channels=1,
    nclasses=2,
    data_num="7_1",  # {7; 10; 15; 25; 50}_[[0,10]]
    e_name="PPT",  # {101a; 101b; PPT}
    nb_crop_per_image=64,
    nuclei=False,  # if false, will be nucleoli
    batch_size=8,  # if distributed training : batch size per gpu
    normalize=True,
    random_rescale=False,
    downsize=True,
    # If True, images are cropped to [dim_crop*dim_crop] then downsized to [dimensions_input*dimensions_input]
    # else images are directly cropped to [dimensions_input*dimensions_input]
    dim_crop=2048,
    dimensions_input=512,
    keep_empty_output_prob=0.001,
    ### MISC
    base_lr=0.00006,
    epochs=750,
    seed=42,
    loss_depth_init=0,
    train_data_path=None,
    train_masks_path=None,
    val_data_path=None,
    val_masks_path=None,
))

random.seed(HPP_DEFAULT.seed)
np.random.seed(HPP_DEFAULT.seed)
torch.manual_seed(HPP_DEFAULT.seed)
torch.cuda.manual_seed(HPP_DEFAULT.seed)

def poly_lr_linear_warmup(base_lr, iter, max_iter):
    if iter < 1500:
        slope = base_lr / 1500
        lr = slope * iter
    else:
        lr = base_lr * (1 - iter / max_iter)
    return lr


def add_weight_decay(net, l2_value=0.01, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name.split(".")[-1] in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


if __name__ == "__main__":
    world_size = 1
    global_rank = 0
    nuclei_string = "nuclei" if HPP_DEFAULT.nuclei else "nucleoli"
    run = wandb.init(project="DTC", config=HPP_DEFAULT, settings=wandb.Settings(start_method="fork"),
                     tags=["Senformer", "batch={}".format(world_size * HPP_DEFAULT.batch_size), "n_gpu={}".format(world_size),
                           "downsize={}".format(HPP_DEFAULT.downsize), nuclei_string])
    model_dir = os.path.join("./models", run.id)
    Path(model_dir).mkdir(parents=True, exist_ok=False)
    RUN_ID = run.id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("init successful")
    base_lr = HPP_DEFAULT.base_lr
    model = SenFormer(num_heads=8, branch_depth=6, in_chans=1, num_classes=HPP_DEFAULT.nclasses, mlp_ratio=4.,
                      qkv_bias=True,
                      qk_scale=None, drop=0.,
                      attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1.e-15,
                      align_corners=False).train().to(device)


    d_train = SegmentationDataset(HPP_DEFAULT, HPP_DEFAULT.train_data_path, HPP_DEFAULT.train_masks_path, train=True, pad_mask=True)
    d_val = SegmentationDataset(HPP_DEFAULT, HPP_DEFAULT.val_data_path, HPP_DEFAULT.val_masks_path, train=False, pad_mask=True)
    train_loader = DataLoader(d_train, HPP_DEFAULT.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(d_val, HPP_DEFAULT.batch_size, pin_memory=True)

    params = add_weight_decay(model,
                              skip_list=["absolute_pos_embed", "relative_position_bias_table", "norm", "queries"])
    optimizer = optim.AdamW(params, lr=HPP_DEFAULT.base_lr, betas=(0.9, 0.999))
    lr_ = HPP_DEFAULT.base_lr
    best_loss = math.inf
    best_dice = 0
    max_iter = len(train_loader) * HPP_DEFAULT.epochs
    iter = 0
    for epoch_num in range(HPP_DEFAULT.epochs):
        to_log = {}
        losses_val, dices_val = [], []
        losses_train, dices_train, = [], []
        # l_images for labeled images, ul_images for unlabeled images
        for i_batch, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            images, targets = images.to(device), targets[0].to(device)

            outputs = model(images)

            # calculate the loss
            loss_seg_dice = model.losses(outputs, targets.long())

            loss = loss_seg_dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, norm_type=2)
            optimizer.step()
            pred = F.interpolate(outputs[1], size=targets.shape[2:], mode="bilinear")
            pred = torch.argmax(pred, dim=1)
            dc = metrics.dice(pred, targets)

            losses_train.append(loss.item())
            dices_train.append(dc.item())

            # update lr
            lr_ = poly_lr_linear_warmup(base_lr, iter, max_iter)
            for g in optimizer.param_groups:
                g['lr'] = lr_
            iter += 1

        model.eval()
        with torch.no_grad():
            for i_batch, (images, targets) in enumerate(val_loader):
                images, targets = images.to(device), targets[0].to(device)

                outputs = model(images)

                # calculate the loss
                loss_seg_dice = model.losses(outputs, targets.long())

                loss = loss_seg_dice

                pred = F.interpolate(outputs[1], size=targets.shape[2:], mode="bilinear")
                pred = torch.argmax(pred, dim=1)
                dc = metrics.dice(pred, targets)

                losses_val.append(loss.item())
                dices_val.append(dc.item())
        model.train()

        # log
        loss_val = statistics.mean(losses_val)
        dice_val = statistics.mean(dices_val)
        to_log["lr"] = lr_
        to_log["train/loss"] = statistics.mean(losses_train)
        to_log["train/dice"] = statistics.mean(dices_train)
        to_log["val/loss"] = loss_val
        to_log["val/dice"] = dice_val

        # save model
        if loss_val < best_loss:
            best_loss = loss_val
            to_log["best_loss"] = best_loss
            # save model
            torch.save({
                "epoch": epoch_num,
                "state_dict": model.state_dict(),
            }, os.path.join(model_dir, "model_best_val.pt"))
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
