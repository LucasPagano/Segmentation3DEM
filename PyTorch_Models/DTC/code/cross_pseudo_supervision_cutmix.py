import math
import os
import statistics
import sys
import numpy
import wandb
from torch import nn


sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
from pathlib import Path
from PyTorch_Models.CEECNET.utils import Dotdict, detach_to_numpy, wb_mask

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from PyTorch_Models.DTC.code.utils.mask_gen import BoxMaskGenerator
from PyTorch_Models.CEECNET.dset import UnsupervisedSegmentationDataset, SegmentationDataset
from PyTorch_Models.CEECNET.eval import run_eval
from networks.unet import SmallUNet

from utils import losses, metrics

HPP_DEFAULT = Dotdict(dict(
    ### CPS
    cps_weight=1,
    base_lr=0.001,
    start_channels=16,
    ### DATASET AND DATALOADER
    in_channels=1,
    nclasses=1,
    data_num="7_1",  # {7; 10; 15; 25; 50}_[[0,10]]
    e_name="101b",  # {101a; 101b; PPT}
    nb_crop_per_image=64,
    nuclei=True,  # if false, will be nucleoli
    batch_size=32,
    normalize=True,
    random_rescale=False,
    downsize=True,
    # If True, images are cropped to [dim_crop*dim_crop] then downsized to [dimensions_input*dimensions_input]
    # else images are directly cropped to [dimensions_input*dimensions_input]
    dim_crop=2048,
    dimensions_input=512,
    keep_empty_output_prob=0.001,
    ### MISC
    epochs=500,
    seed=42,
    loss_depth_init=0,
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
    model = SmallUNet(1, 1, HPP_DEFAULT.start_channels).train().to(device)
    model2 = SmallUNet(1, 1, HPP_DEFAULT.start_channels).train().to(device)

    nuclei_string = "nuclei" if HPP_DEFAULT.nuclei else "nucleoli"
    img_string = "images" if HPP_DEFAULT.e_name == "PPT" or HPP_DEFAULT.nuclei else "nucleolus_images"
    msk_string = "nucleus_mask" if HPP_DEFAULT.nuclei else "nucleolus_mask"
    e_name = HPP_DEFAULT.e_name
    data_num = HPP_DEFAULT.data_num
    # if train data num is X_Y val data num will be X_Y+1, eg : train 25_3 -> val 25_4, 25_9 -> 25_1
    val_data_num = data_num.split("_")[0] + "_" + str((int(data_num.split("_")[-1]) + 1) % 10)
    val_data_num = val_data_num[:-1] + "1" if val_data_num[-1] == "0" else val_data_num
    val_data_num = "25_9" if data_num[-1] in ["1", "2", "3", "4", "5"] else "25_1"
    train_data_path = "/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/new_10/" + e_name + "/" + data_num + "/" + img_string + "/1"
    train_masks_path = "/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/new_10/" + e_name + "/" + data_num + "/" + msk_string + "/1"
    val_data_path = "/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/new_10/" + e_name + "/" + val_data_num + "/" + img_string + "/1"
    val_masks_path = "/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/new_10/" + e_name + "/" + val_data_num + "/" + msk_string + "/1"
    classes_out = ["__background__", nuclei_string]
    d_train = UnsupervisedSegmentationDataset(HPP_DEFAULT, train_data_path, train_masks_path, train=True)
    d_val = SegmentationDataset(HPP_DEFAULT, val_data_path, val_masks_path, train=False)
    train_loader = DataLoader(d_train, fake_batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(d_val, HPP_DEFAULT.batch_size, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=HPP_DEFAULT.base_lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=HPP_DEFAULT.base_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15, cooldown=15, threshold=1e-5, min_lr=base_lr * 1e-2, verbose=True)
    lr_ = base_lr
    best_loss = math.inf
    best_dice = 0
    iter = 0
    max_iter = len(train_loader) * HPP_DEFAULT.epochs
    boxmix_gen = BoxMaskGenerator(prop_range=(0.25, 0.5), random_aspect_ratio=True, prop_by_area=True, within_bounds=True, invert=True)
    for epoch_num in range(HPP_DEFAULT.epochs):
        to_log = {}
        losses_val, dices_val = [], []
        losses_train, cps1_, cps2_, seg1, seg2, dices_train, dices_train2 = [], [], [], [], [], [], []
        # l_images for labeled images, ul_images for unlabeled images
        for i_batch, (l_images, targets, ul_images) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer2.zero_grad()
            if diff > 0: # remove last ul to get to odd batch size if needed
                ul_images = ul_images[:-1]
            split_size_l = int(l_images.size(0) / 2) if l_images.size(0) % 2 == 0 else int(l_images.size(0) // 2 + 1)
            split_size_u = int(ul_images.size(0) / 2) if ul_images.size(0) % 2 == 0 else int(ul_images.size(0) // 2 + 1)
            l_1, l_2 = torch.split(l_images, split_size_l, dim=0)
            ul_1, ul_2 = torch.split(ul_images, split_size_u, dim=0)
            if l_1.size(0) != l_2.size(0):
                l_1 = l_1[:-1]
            if ul_1.size(0) != ul_2.size(0):
                ul_1 = ul_1[:-1]
            ul_1, ul_2 = torch.cat((l_1, ul_1), dim=0), torch.cat((l_2, ul_2), dim=0)
            batch_mix_masks = boxmix_gen.generate_params(n_masks=ul_1.size(0), mask_shape=(ul_1.size(2), ul_1.size(3)))
            batch_mix_masks = torch.as_tensor(batch_mix_masks).type(torch.float32).to(device, non_blocking=True)
            l_batch, targets = l_images.to(device, non_blocking=True), targets[0].cuda(non_blocking=True)
            ul_1, ul_2 = ul_1.to(device, non_blocking=True), ul_2.to(device, non_blocking=True)
            unsup_imgs_mixed = ul_1 * (1 - batch_mix_masks) + ul_2 * batch_mix_masks
            with torch.no_grad():
                # Estimate the pseudo-label with branch#1 & supervise branch#2
                logits_u0_tea_1 = model(ul_1)
                logits_u1_tea_1 = model(ul_2)
                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # Estimate the pseudo-label with branch#2 & supervise branch#1
                logits_u0_tea_2 = model2(ul_1)
                logits_u1_tea_2 = model2(ul_2)
                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()

            # Mix predictions using same mask
            # It makes no difference whether we do this with logits or probabilities as
            # the mask pixels are either 1 or 0
            logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            ps_label_1 = torch.where(logits_cons_tea_1>0.5,1,0)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            ps_label_2 = torch.where(logits_cons_tea_2>0.5,1,0)
            ps_label_2 = ps_label_2.long()

            # Get student#1 prediction for mixed image
            logits_cons_stu_1 = model(unsup_imgs_mixed)
            # Get student#2 prediction for mixed image
            logits_cons_stu_2 = model2(unsup_imgs_mixed)
            cps1 = losses.dice_loss(logits_cons_stu_1, ps_label_2)
            cps2 = losses.dice_loss(logits_cons_stu_2, ps_label_1)
            cps_loss = cps1 + cps2
            cps_loss *= lamdba_scaling(epoch_num)

            # Supervision loss
            outputs_sup, outputs_sup2 = model(l_batch),  model2(l_batch)
            loss_dice = losses.dice_loss(outputs_sup.squeeze(), targets.squeeze() == 1)
            loss_dice2 = losses.dice_loss(outputs_sup2.squeeze(), targets.squeeze() == 1)
            loss_dice_total = loss_dice + loss_dice2
            loss = cps_loss + loss_dice_total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)

            optimizer.step()
            optimizer2.step()

            cps1_.append(cps1.item())
            cps2_.append(cps2.item())
            losses_train.append(loss.item())
            dices_train.append(loss_dice.item())
            dices_train2.append(loss_dice2.item())

            # update lr
            lr_ = poly_lr_linear_warmup(base_lr, iter, max_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            iter += 1

            # log imgs
            if i_batch == 0:
                log_amount = 2
                for i in range(log_amount):
                    img = detach_to_numpy(unsup_imgs_mixed[i].squeeze())
                    msk = torch.where(ps_label_2[i].squeeze() > 0.5, 1, 0)
                    msk = detach_to_numpy(msk).astype(numpy.uint8)
                    pred_msk = torch.where(logits_cons_stu_1[i].squeeze() > 0.5, 1, 0)
                    pred_msk = detach_to_numpy(pred_msk).astype(numpy.uint8)
                    to_log_img = wb_mask(bg_img=img, pred_mask=pred_msk, true_mask=msk, classes_out=classes_out)
                    to_log["images/epoch{}_{}".format(epoch_num, i)] = to_log_img
        model.eval()
        with torch.no_grad():
            for i_batch, (images, targets) in enumerate(val_loader):
                l_batch, targets = images.to(device), targets[0].to(device)
                outputs_sup= model(l_batch)
                dc = metrics.dice(torch.where(outputs_sup.squeeze() > 0.5, 1, 0), targets)

                dices_val.append(dc.item())
        model.train()

        dice_val = statistics.mean(dices_val)

        # log
        to_log["lr"] = lr_
        to_log["lambda"] = lamdba_scaling(epoch_num)
        to_log["train/loss"] = statistics.mean(losses_train)
        to_log["train/cps1"] = statistics.mean(cps1_)
        to_log["train/cps2"] = statistics.mean(cps2_)
        to_log["train/dice"] = statistics.mean(dices_train)
        to_log["train/dice2"] = statistics.mean(dices_train2)
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