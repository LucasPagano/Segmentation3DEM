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
from PyTorch_Models.CEECNET.utils import Dotdict, detach_to_numpy, wb_mask
import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader

from PyTorch_Models.CEECNET.dset import UnsupervisedSegmentationDataset
from networks.unet import UNet

from utils import ramps, losses, metrics
from utils.util import compute_sdf

HPP_DEFAULT = Dotdict(dict(
    ### DTC
    beta=0.3,
    base_lr=0.01,
    max_iterations=150000,
    consistency=1.0,
    consistency_rampup=40.0,
    consistency_weight=0.1,
    consistency_type="mse",
    ### DATASET AND DATALOADER
    in_channels=1,
    nclasses=1,
    data_num="7_1",  # {7; 10; 15; 25; 50}_[[0,10]]
    e_name="101b",  # {101a; 101b; PPT}
    nb_crop_per_image=64,
    nuclei=False,  # if false, will be nucleoli
    batch_size=4,  # if distributed training : batch size per gpu
    normalize=True,
    random_rescale=False,
    downsize=True,
    # If True, images are cropped to [dim_crop*dim_crop] then downsized to [dimensions_input*dimensions_input]
    # else images are directly cropped to [dimensions_input*dimensions_input]
    dim_crop=2048,
    dimensions_input=512,
    keep_empty_output_prob=0.001,
    ### MISC
    epochs=50,
    seed=42,
    loss_depth_init=0,
))

random.seed(HPP_DEFAULT.seed)
np.random.seed(HPP_DEFAULT.seed)
torch.manual_seed(HPP_DEFAULT.seed)
torch.cuda.manual_seed(HPP_DEFAULT.seed)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return HPP_DEFAULT.consistency * ramps.sigmoid_rampup(epoch, HPP_DEFAULT.consistency_rampup)


if __name__ == "__main__":
    world_size = 1
    global_rank = 0
    nuclei_string = "nuclei" if HPP_DEFAULT.nuclei else "nucleoli"
    run = wandb.init(project="DTC", config=HPP_DEFAULT, settings=wandb.Settings(start_method="fork"),
                     tags=["batch={}".format(world_size * HPP_DEFAULT.batch_size), "n_gpu={}".format(world_size),
                           "downsize={}".format(HPP_DEFAULT.downsize), nuclei_string])
    model_dir = os.path.join("./models", run.id)
    Path(model_dir).mkdir(parents=True, exist_ok=False)
    RUN_ID = run.id
    print("init successful")
    base_lr = HPP_DEFAULT.base_lr
    model = UNet(1, 1).cuda().train()
    wandb.watch(model, log="None")

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
    d_val = UnsupervisedSegmentationDataset(HPP_DEFAULT, val_data_path, val_masks_path, train=False)
    trainloader = DataLoader(d_train, HPP_DEFAULT.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(d_val, HPP_DEFAULT.batch_size, pin_memory=True)
    optimizer = optim.SGD(model.parameters(), lr=HPP_DEFAULT.base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    if HPP_DEFAULT.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif HPP_DEFAULT.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, HPP_DEFAULT.consistency_type

    iter_num = 0
    lr_ = HPP_DEFAULT.base_lr
    best_loss = math.inf
    best_dice = 0

    for epoch_num in range(HPP_DEFAULT.epochs):
        to_log = {}
        losses_val, dices_val, consistencies_val, hausdorffs_val, segs_val = [], [], [], [], []
        losses_train, dices_train, consistencies_train, hausdorffs_train, segs_train = [], [], [], [], []
        # l_images for labeled images, ul_images for unlabeled images
        for i_batch, (l_images, targets, ul_images) in enumerate(trainloader):
            optimizer.zero_grad()
            ul_batch, l_batch, targets = ul_images.cuda(), l_images.cuda(), targets[0].cuda()
            volume_batch = torch.cat((l_batch, ul_batch), dim=0)

            outputs_tanh, outputs = model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)

            # calculate the loss
            with torch.no_grad():
                gt_dis = compute_sdf(targets[:].cpu(
                ).numpy(), outputs[:len(l_batch), 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_sdf = mse_loss(outputs_tanh[:len(l_batch), 0, ...], gt_dis)
            loss_seg = ce_loss(
                outputs[:len(l_batch), 0, ...].squeeze(), targets.squeeze().float())
            loss_seg_dice = losses.dice_loss(
                outputs_soft[:len(l_batch), 0, :, :], targets.squeeze() == 1)
            dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)

            consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
            supervised_loss = loss_seg_dice + HPP_DEFAULT.beta * loss_sdf
            consistency_weight = get_current_consistency_weight(epoch_num)

            loss = supervised_loss + consistency_weight * consistency_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dc = metrics.dice(torch.where(outputs_soft[:len(l_batch)] > 0.5, 1, 0), targets)

            losses_train.append(loss.item())
            dices_train.append(dc.item())
            hausdorffs_train.append(loss_sdf.item())
            segs_train.append(supervised_loss.item())
            consistencies_train.append(consistency_loss.item())
            iter_num += 1
            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

        model.eval()
        with torch.no_grad():
            # l_images for labeled images, ul_images for unlabeled images
            for i_batch, (l_images, targets, ul_images) in enumerate(val_loader):
                ul_batch, l_batch, targets = ul_images.cuda(), l_images.cuda(), targets[0].cuda()
                volume_batch = torch.cat((l_batch, ul_batch), dim=0)

                outputs_tanh, outputs = model(volume_batch)
                outputs_soft = torch.sigmoid(outputs)

                gt_dis = compute_sdf(targets[:].cpu().numpy(), outputs[:len(l_batch), 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
                loss_sdf = mse_loss(outputs_tanh[:len(l_batch), 0, ...], gt_dis)
                loss_seg = ce_loss(
                    outputs[:len(l_batch), 0, ...].squeeze(), targets.squeeze().float())
                loss_seg_dice = losses.dice_loss(
                    outputs_soft[:len(l_batch), 0, :, :], targets.squeeze() == 1)
                dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)

                consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
                supervised_loss = loss_seg_dice + HPP_DEFAULT.beta * loss_sdf
                consistency_weight = get_current_consistency_weight(epoch_num)

                loss = supervised_loss + consistency_weight * consistency_loss
                dc = metrics.dice(torch.where(outputs_soft[:len(l_batch)] > 0.5, 1, 0), targets)

                losses_val.append(loss.item())
                dices_val.append(dc.item())
                hausdorffs_val.append(loss_sdf.item())
                segs_val.append(supervised_loss.item())
                consistencies_val.append(consistency_loss.item())

                # log images
                if i_batch == 0:
                    for j, img in enumerate(l_batch):
                        if j < 2:
                            img = detach_to_numpy(img.squeeze())
                            gt_msk = detach_to_numpy(targets[j].squeeze()).astype(np.uint8)
                            pred_msk = detach_to_numpy(torch.where(outputs_soft[j] > 0.5, 1, 0).squeeze()).astype(np.uint8)
                            to_log_img = wb_mask(bg_img=img, pred_mask=pred_msk, true_mask=gt_msk, classes_out=classes_out)
                            to_log["images/epoch{}_{}".format(epoch_num, j)] = to_log_img

                            gt_lsf = wandb.Image(detach_to_numpy(gt_dis[j].squeeze()))
                            pred_lsf = wandb.Image(detach_to_numpy(outputs_tanh[j].squeeze()))
                            to_log["lsf/epoch{}_{}_gt".format(epoch_num, j)] = gt_lsf
                            to_log["lsf/epoch{}_{}_pred".format(epoch_num, j)] = pred_lsf
            model.train()

            # log
            loss_val = statistics.mean(losses_val)
            dice_val = statistics.mean(dices_val)
            to_log["lr"] = lr_
            to_log["consistency_weight"] = consistency_weight
            to_log["train/loss"] = statistics.mean(losses_train)
            to_log["train/dice"] = statistics.mean(dices_train)
            to_log["train/consistency_loss"] = statistics.mean(consistencies_train)
            to_log["train/hausdorff_loss"] = statistics.mean(hausdorffs_train)
            to_log["train/seg_loss"] = statistics.mean(segs_train)
            to_log["val/loss"] = loss_val
            to_log["val/dice"] = dice_val
            to_log["val/consistency_loss"] = statistics.mean(consistencies_val)
            to_log["val/hausdorff_loss"] = statistics.mean(hausdorffs_val)
            to_log["val/seg_loss"] = statistics.mean(segs_val)

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

        if iter_num >= HPP_DEFAULT.max_iterations:
            break
