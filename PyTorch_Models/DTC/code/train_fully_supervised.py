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
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from PyTorch_Models.CEECNET.dset import SegmentationDataset
from networks.unet import SmallUNet

from utils import ramps, losses, metrics

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
    nuclei=True,  # if false, will be nucleoli
    batch_size=8,  # if distributed training : batch size per gpu
    normalize=True,
    random_rescale=False,
    downsize=True,
    # If True, images are cropped to [dim_crop*dim_crop] then downsized to [dimensions_input*dimensions_input]
    # else images are directly cropped to [dimensions_input*dimensions_input]
    dim_crop=2048,
    dimensions_input=700,
    keep_empty_output_prob=0.001,
    ### MISC
    epochs=500,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("init successful")
    base_lr = HPP_DEFAULT.base_lr
    model = SmallUNet(1, 1).train().to(device)
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
    d_train = SegmentationDataset(HPP_DEFAULT, train_data_path, train_masks_path, train=True)
    d_val = SegmentationDataset(HPP_DEFAULT, val_data_path, val_masks_path, train=False)
    train_loader = DataLoader(d_train, HPP_DEFAULT.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(d_val, HPP_DEFAULT.batch_size, pin_memory=True)
    optimizer = optim.SGD(model.parameters(), lr=HPP_DEFAULT.base_lr,
                          momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    lr_ = HPP_DEFAULT.base_lr
    best_loss = math.inf
    best_dice = 0

    for epoch_num in range(HPP_DEFAULT.epochs):
        to_log = {}
        losses_val, dices_val = [], []
        losses_train, dices_train, = [], []
        # l_images for labeled images, ul_images for unlabeled images
        for i_batch, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            images, targets = images.to(device), targets[0].to(device)

            outputs_soft = model(images)

            # calculate the loss
            loss_seg_dice = losses.dice_loss(outputs_soft.squeeze(), targets.squeeze() == 1)

            loss = loss_seg_dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dc = metrics.dice(torch.where(outputs_soft.squeeze() > 0.5, 1, 0), targets)

            losses_train.append(loss.item())
            dices_train.append(dc.item())
            iter_num += 1

        model.eval()
        with torch.no_grad():
            for i_batch, (images, targets) in enumerate(val_loader):
                images, targets = images.to(device), targets[0].to(device)

                outputs_soft = model(images)

                # calculate the loss
                loss_seg_dice = losses.dice_loss(outputs_soft.squeeze(), targets.squeeze() == 1)

                loss = loss_seg_dice

                dc = metrics.dice(torch.where(outputs_soft.squeeze() > 0.5, 1, 0), targets)

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

        if iter_num >= HPP_DEFAULT.max_iterations:
            break
