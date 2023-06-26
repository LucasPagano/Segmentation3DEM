import math
import os
import statistics
import sys
from pathlib import Path

import wandb
import numpy as np
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler

from PyTorch_Models.CEECNET.eval import run_eval
from PyTorch_Models.CEECNET.ceecnet import XNetSegmentation
from PyTorch_Models.CEECNET.dset import SegmentationDataset
from PyTorch_Models.CEECNET.loss import MtskLoss
from PyTorch_Models.CEECNET.utils import Dotdict, wb_mask, detach_to_numpy, dice_coeff, get_smp_model, dice_coef_loss, rand_bbox

HPP_DEFAULT = Dotdict(dict(
    ### MODEL
    ceecnet=True,  # if True, ceecnet models will be used, else "segmentation models pytorch (smp)" lib will be used
    ### SMP
    model_smp="Unet++",  # {Unet; Unet++; Linknet; FPN; DeepLabV3+}
    encoder_smp="resnet34",  # {resnet{18;34;50;101;152}; vgg{11; 13; 16; 19}{;_bn}; timm-regnet{x;y}_{040; 120; 320}
    activation_smp="sigmoid",
    ### CEECNET
    nfilters_init=16,
    depth=4,
    widths=[1],
    psp_depth=4,
    verbose=True,
    norm_type="GroupNorm",
    upFuse=False,
    norm_groups=4,
    nheads_start=2,
    model="CEECNetV2",  # {CEECNetV1; CEECNetV2; FracTALResNet}
    ftdepth=5,
    ### DATASET AND DATALOADER
    in_channels=1,
    nclasses=1,
    data_num="15_1",  # {7; 10; 15; 25; 50}_[[0,10]]
    e_name="101a",  # {101a; 101b; PPT}
    nb_crop_per_image=64,
    nuclei=False,  # if false, will be nucleoli
    batch_size=3,  # if distributed training : batch size per gpu
    normalize=True,
    random_rescale=False,
    downsize=True,
    # If True, images are cropped to [dim_crop*dim_crop] then downsized to [dimensions_input*dimensions_input]
    # else images are directly cropped to [dimensions_input*dimensions_input]
    dim_crop=2048,
    dimensions_input=512,
    shuffle=True,
    keep_empty_output_prob=0.001,
    ### MISC
    train_data_path=None,
    train_masks_path=None,
    val_data_path=None,
    val_masks_path=None,
    cutmix=False,
    cutmix_prob=0.5,
    beta=1.0,
    distributed_training=True,
    epochs=500,
    seed=42,
    loss_depth_init=0,
))


def train(rank, world_size):
    config = HPP_DEFAULT
    nuclei_string = "nuclei" if config.nuclei else "nucleoli"
    model_string = config.model if config.ceecnet else config.model_smp
    if config.distributed_training:
        global_rank = int((os.environ["SLURM_PROCID"]))
        # initialize the process group
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
        torch.cuda.set_device(rank)
        if global_rank == 0:
            run = wandb.init(project="ceecnet", config=config, settings=wandb.Settings(start_method="fork"),
                             tags=["batch={}".format(world_size * config.batch_size), "n_gpu={}".format(world_size), model_string,
                                   "downsize={}".format(config.downsize), nuclei_string])
            model_dir = os.path.join("./models", run.id)
            Path(model_dir).mkdir(parents=True, exist_ok=False)
            RUN_ID = run.id

    else:
        world_size = 1
        global_rank = 0
        run = wandb.init(project="ceecnet", config=config, settings=wandb.Settings(start_method="fork"),
                         tags=["batch={}".format(world_size * config.batch_size), "n_gpu={}".format(world_size), model_string,
                               "downsize={}".format(config.downsize), nuclei_string])
        model_dir = os.path.join("./models", run.id)
        Path(model_dir).mkdir(parents=True, exist_ok=False)
        config = wandb.config
        RUN_ID = run.id

    # learning rate scaled to 5e-4 for a batch size of 256
    if config.ceecnet:
        learning_rate = (world_size * config.batch_size) * 5e-4 / 256
    else:
        learning_rate = (world_size * config.batch_size) * 10e-5 / 256
    base_lr = learning_rate

    # only log on rank zero
    logging = global_rank == 0

    print("Init successful!")
    print("Global rank : {}".format(global_rank))

    # random init
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device = rank if config.distributed_training else device
    if config.ceecnet:
        model = XNetSegmentation(config).to(device)
        loss = MtskLoss(config.loss_depth_init)
        loss_depth = config.loss_depth_init
    else:
        model = get_smp_model(config).to(device)
        loss = dice_coef_loss

    if config.distributed_training:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], output_device=rank)
    if logging:
        wandb.watch(model, log="None")

    train_dataset = SegmentationDataset(config, config.train_data_path, config.train_masks_path, train=True)
    val_dataset = SegmentationDataset(config, config.val_data_path, config.val_masks_path, train=False)
    # let the sampler handle the shuffling if used
    if config.distributed_training:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        sampler = None
        val_sampler = None
        shuffle = config.shuffle
    epoch_log_step = config.epochs // 10 if config.epochs > 10 else 5
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle, sampler=sampler, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, config.batch_size, shuffle=False, sampler=val_sampler, pin_memory=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # reduce learning rate at most twice
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, cooldown=15, threshold=1e-5,
                                  min_lr=base_lr * 1e-2,
                                  verbose=True)
    best_val_loss = torch.tensor(math.inf)
    best_dice = torch.tensor(0)
    avg_val_loss = torch.tensor(0).to(device)
    avg_dice = torch.tensor(0).to(device)
    print("STARTING TRAINING")
    for epoch in range(config.epochs):
        if config.distributed_training:
            sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        dice_coeffs = []
        losses_train, losses_val = [], []
        losses_train_segm, losses_train_bound, losses_train_dist = [], [], []
        losses_val_segm, losses_val_bound, losses_val_dist = [], [], []
        to_log = {}
        # train step
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            images, targets = images.to(device), [tensor.to(device) for tensor in targets]
            if config.cutmix and np.random.rand(1) < config.cutmix_prob:
                # generate mixed sample
                # !! only works with ceecnet=False
                images = images[:images.size(0) // 2, :, :, :]
                targets = targets[0][:images.size(0), :, :, :]
                lam = np.random.beta(config.beta, config.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                outs = model(images)
                train_loss = loss(outs, target_a) * lam + loss(outs, target_b) * (1. - lam)
            else:
                outs = model(images)
                if config.ceecnet:
                    loss_segm, loss_bound, loss_dist = loss(outs, targets)
                    losses_train_segm.append(loss_segm.item())
                    losses_train_bound.append(loss_bound.item())
                    losses_train_dist.append(loss_dist.item())
                    train_loss = (loss_segm + loss_bound + loss_dist) / 3.0
                else:
                    train_loss = loss(outs, targets[0])

            losses_train.append(train_loss.item())
            train_loss.backward()
            optimizer.step()

        # val step, distributed forward and gather to log only on rank 0
        with torch.no_grad():
            model.eval()
            for i, (images, targets) in enumerate(val_loader):
                images, targets = images.to(device), [tensor.to(device) for tensor in targets]
                outs = model(images)

                # images logging
                if i == 0 and epoch % epoch_log_step == 0 and logging:
                    print("Epoch {}".format(epoch))
                    print("logging images... ")
                    for j, img in enumerate(images):
                        if j < 5:
                            img = detach_to_numpy(img.squeeze())
                            msk = detach_to_numpy(targets[0][j].squeeze()).astype(np.uint8)
                            if config.ceecnet:
                                to_log["boundaries/epoch{}_{}".format(epoch, j)] = wandb.Image(detach_to_numpy(outs[1][j].squeeze()))
                                to_log["boundaries/epoch{}_{}_gt".format(epoch, j)] = wandb.Image(detach_to_numpy(targets[1][j].squeeze()))
                                to_log["distance_maps/epoch{}_{}".format(epoch, j)] = wandb.Image(detach_to_numpy(outs[2][j].squeeze()))
                                to_log["distance_maps/epoch{}_{}_gt".format(epoch, j)] = wandb.Image(detach_to_numpy(targets[2][j].squeeze()))
                            prd_msk = outs[0][j] if config.ceecnet else outs[j]
                            pred_masks = detach_to_numpy(torch.where(prd_msk > 0.5, 1, 0).squeeze()).astype(np.uint8)
                            to_log_img = wb_mask(bg_img=img, pred_mask=pred_masks, true_mask=msk, classes_out=classes_out)
                            to_log["images/epoch{}_{}".format(epoch, j)] = to_log_img

                # metrics computation
                if config.ceecnet:
                    loss_segm, loss_bound, loss_dist = loss(outs, targets)
                    val_loss = (loss_segm + loss_bound + loss_dist) / 3.0
                    pred = torch.where(outs[0] > 0.5, 1, 0)
                else:
                    val_loss = loss(outs, targets[0])
                    pred = torch.where(outs > 0.5, 1, 0)
                dc = dice_coeff(pred, targets[0])
                if config.distributed_training:
                    # average all tensors before logging
                    # create place holders
                    placeholders_segm, placeholders_bound, placeholders_dist, placeholders_val_loss, placeholders_dice = (
                        [torch.zeros_like(dc).to(device) for _ in range(world_size)] for _ in range(5))
                    # gather values
                    if config.ceecnet:
                        torch.distributed.all_gather(placeholders_dist, loss_dist)
                        torch.distributed.all_gather(placeholders_bound, loss_bound)
                        torch.distributed.all_gather(placeholders_segm, loss_segm)
                    torch.distributed.all_gather(placeholders_val_loss, val_loss)
                    torch.distributed.all_gather(placeholders_dice, dc)
                    # average and log values on rank 0 only
                    if logging:
                        if config.ceecnet:
                            loss_segm = statistics.mean([tensor.item() for tensor in placeholders_segm])
                            loss_dist = statistics.mean([tensor.item() for tensor in placeholders_dist])
                            loss_bound = statistics.mean([tensor.item() for tensor in placeholders_bound])
                            losses_val_segm.append(loss_segm)
                            losses_val_dist.append(loss_dist)
                            losses_val_bound.append(loss_bound)
                        val_loss = statistics.mean([tensor.item() for tensor in placeholders_val_loss])
                        dc = statistics.mean([tensor.item() for tensor in placeholders_dice])
                        dice_coeffs.append(dc)
                        losses_val.append(val_loss)
                else:
                    if config.ceecnet:
                        losses_val_segm.append(loss_segm.item())
                        losses_val_dist.append(loss_dist.item())
                        losses_val_bound.append(loss_bound.item())
                    dice_coeffs.append(dc.item())
                    losses_val.append(val_loss.item())
            model.train()

        if logging:
            avg_val_loss = torch.tensor(statistics.mean(losses_val)).to(device)
            avg_dice = torch.tensor(statistics.mean(dice_coeffs)).to(device)
            if avg_dice > best_dice:
                best_dice = avg_dice
                to_log["best_dice"] = avg_dice
                state_dict = model.module.state_dict() if config.distributed_training else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "state_dict": state_dict,
                }, os.path.join(model_dir, "model_best_dice.pt"))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                to_log["best_val_loss"] = avg_val_loss
                state_dict = model.module.state_dict() if config.distributed_training else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "state_dict": state_dict,
                }, os.path.join(model_dir, "model_best_val.pt"))

        if config.distributed_training:
            # broadcast values for scheduler step
            torch.distributed.broadcast(avg_dice, src=0)
            torch.distributed.broadcast(avg_val_loss, src=0)

        # learning rate scheduler step
        scheduler.step(avg_val_loss)

        # if learning rate was updated
        if optimizer.param_groups[0]["lr"] != learning_rate:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, cooldown=15, threshold=1e-5,
                                          min_lr=base_lr * 1e-2,
                                          verbose=True)
            # log
            learning_rate = optimizer.param_groups[0]["lr"]
            if config.ceecnet:
                # change tanimoto loss depth
                print("changed loss depth : {} -> {}".format(loss_depth, loss_depth + 10))
                loss_depth += 10
                loss = MtskLoss(loss_depth)

        # logging
        if logging:
            # most important metrics
            to_log["metrics/loss_val"] = avg_val_loss
            to_log["metrics/loss_train"] = statistics.mean(losses_train)
            to_log["metrics/dice"] = avg_dice
            to_log["metrics/learning_rate"] = learning_rate

            if config.ceecnet:
                # train
                to_log["losses/segm"] = statistics.mean(losses_train_segm)
                to_log["losses/dist"] = statistics.mean(losses_train_dist)
                to_log["losses/bound"] = statistics.mean(losses_train_bound)

                # val
                to_log["losses/segm_val"] = statistics.mean(losses_val_segm)
                to_log["losses/dist_val"] = statistics.mean(losses_val_dist)
                to_log["losses/bound_val"] = statistics.mean(losses_val_bound)

                to_log["metrics/tanimoto_loss_depth"] = loss_depth

                # gammas
                mdl = model.module if config.distributed_training else model
                for k, v in mdl.get_gammas().items():
                    to_log["gammas/{}".format(k)] = v

            wandb.log(to_log)

    # cleanup
    if config.distributed_training:
        dist.destroy_process_group()
    if logging:
        wandb.finish()

    # run eval only once
    if global_rank == 0:
        run_eval(run_id=RUN_ID)
        sys.exit()
    else:
        sys.exit()

def main():
    if HPP_DEFAULT.distributed_training:
        local_rank = int(os.environ["SLURM_PROCID"]) % torch.cuda.device_count()
        train(local_rank, int(os.environ["WORLD_SIZE"]))
    else:
        train(None, None)


if __name__ == "__main__":
    main()
