import math
import os
import statistics
import sys
import segmentation_models_pytorch as smp
import wandb
from torch import nn
import torch.nn.functional as F

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
from pathlib import Path
from PyTorch_Models.CEECNET.utils import Dotdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PyTorch_Models.CPS.code.networks.senformer import SenFormer
from PyTorch_Models.CEECNET.dset import UnsupervisedSegmentationDataset, SegmentationDataset
from PyTorch_Models.CEECNET.eval import run_eval
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
    e_name="PPT",  # {101a; 101b; PPT}
    nb_crop_per_image=64,
    nuclei=True,  # if false, will be nucleoli
    batch_size=5,
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


def lamdba_scaling(epoch):
    if epoch <= 5:
        return 1e-5
    elif epoch <= 20:
        return 0.066 * epoch - 0.32
    else:
        return 1


if __name__ == "__main__":
    world_size = 1
    global_rank = 0
    nuclei_string = "nuclei" if HPP_DEFAULT.nuclei else "nucleoli"
    run = wandb.init(project="DTC", config=HPP_DEFAULT, settings=wandb.Settings(start_method="fork"),
                     tags=["Dice_loss", "Senformer", "SSL", "batch={}".format(world_size * HPP_DEFAULT.batch_size), "n_gpu={}".format(world_size),
                           "downsize={}".format(HPP_DEFAULT.downsize), nuclei_string])
    model_dir = os.path.join("./models", run.id)
    Path(model_dir).mkdir(parents=True, exist_ok=False)
    RUN_ID = run.id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("init successful")
    base_lr = HPP_DEFAULT.base_lr
    model = SenFormer(num_heads=8, branch_depth=HPP_DEFAULT.branch_depth, in_chans=1, num_classes=HPP_DEFAULT.nclasses, mlp_ratio=4.,
                      qkv_bias=True,
                      qk_scale=None, drop=0.,
                      attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1.e-15,
                      align_corners=False, swin_tiny=HPP_DEFAULT.swin_tiny).train().to(device)

    model2 = SenFormer(num_heads=8, branch_depth=HPP_DEFAULT.branch_depth, in_chans=1, num_classes=HPP_DEFAULT.nclasses, mlp_ratio=4.,
                       qkv_bias=True,
                       qk_scale=None, drop=0.,
                       attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1.e-15,
                       align_corners=False, swin_tiny=HPP_DEFAULT.swin_tiny).train().to(device)

    fake_batch_size, diff = (HPP_DEFAULT.batch_size // 2 + 1, 1) if HPP_DEFAULT.batch_size % 2 != 0 else (HPP_DEFAULT.batch_size // 2, 0)

    d_train = UnsupervisedSegmentationDataset(HPP_DEFAULT, HPP_DEFAULT.train_data_path, HPP_DEFAULT.train_masks_path, train=True)
    d_val = SegmentationDataset(HPP_DEFAULT, HPP_DEFAULT.val_data_path, HPP_DEFAULT.val_masks_path, train=False)
    train_loader = DataLoader(d_train, fake_batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(d_val, HPP_DEFAULT.batch_size, pin_memory=True)
    params1 = add_weight_decay(model,
                               skip_list=["absolute_pos_embed", "relative_position_bias_table", "norm", "queries"])
    params2 = add_weight_decay(model2,
                               skip_list=["absolute_pos_embed", "relative_position_bias_table", "norm", "queries"])
    optimizer = optim.AdamW(params1, lr=HPP_DEFAULT.base_lr, betas=(0.9, 0.999))
    optimizer2 = optim.AdamW(params2, lr=HPP_DEFAULT.base_lr, betas=(0.9, 0.999))
    loss_dice_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    lr_ = base_lr
    best_loss = math.inf
    best_dice = 0
    iter = 0
    max_iter = len(train_loader) * HPP_DEFAULT.epochs
    eps = 1.e-15
    for epoch_num in range(HPP_DEFAULT.epochs):
        to_log = {}
        losses_val, dices_val = [], []
        losses_train, cps1_ensemble_, cps1_logits_, cps2_ensemble_, cps2_logits_, dices_train_ensemble, dices_train2_ensemble, dices_train_logits, dices_train2_logits = [], [], [], [], [], [], [], [], []
        # l_images for labeled images, ul_images for unlabeled images
        for i_batch, (l_images, targets, ul_images) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer2.zero_grad()
            ul_batch, l_batch, targets = ul_images.to(device), l_images.to(device), targets[0].to(device)
            if diff > 0: # remove last ul to get to odd batch size if needed
                ul_images = ul_images[:-1]

            outsup, outsup2 = model(l_batch), model2(l_batch)
            outunsup, outunsup2 = model(ul_batch), model2(ul_batch)

            # CPS LOSS
            # generate Y1 and Y2
            ensemble1_sup = torch.log(torch.clamp(outsup[1], min=eps))
            ensemble2_sup = torch.log(torch.clamp(outsup2[1], min=eps))
            ensemble1_unsup = torch.log(torch.clamp(outunsup[1], min=eps))
            ensemble2_unsup = torch.log(torch.clamp(outunsup2[1], min=eps))

            outputs_sup = F.interpolate(ensemble1_sup, size=targets.shape[2:], mode="bilinear")
            outputs_unsup = F.interpolate(ensemble1_unsup, size=targets.shape[2:], mode="bilinear")
            outputs_sup2 = F.interpolate(ensemble2_sup, size=targets.shape[2:], mode="bilinear")
            outputs_unsup2 = F.interpolate(ensemble2_unsup, size=targets.shape[2:], mode="bilinear")

            # CPS LOSS
            pred_l = torch.cat([outputs_sup, outputs_unsup], dim=0)
            pred_r = torch.cat([outputs_sup2, outputs_unsup2], dim=0)
            max_l = torch.argmax(pred_l, dim=1)
            max_r = torch.argmax(pred_r, dim=1)
            max_l = max_l.long()
            max_r = max_r.long()
            # ensemble
            cps1_ensemble = loss_dice_fn(pred_l, max_r)
            cps2_ensemble = loss_dice_fn(pred_r, max_l)

            # learners, each supervised with ensemble pred
            logits_l_sup = [F.interpolate(input=outsup[0][i], size=targets.shape[2:], mode='bilinear',
                            align_corners=False) for i in range(len(outsup[0]))]
            logits_l_unsup = [F.interpolate(input=outunsup[0][i], size=targets.shape[2:], mode='bilinear',
                            align_corners=False) for i in range(len(outunsup[0]))]
            logits_l = [torch.cat((logits_l_sup[i], logits_l_unsup[i]), dim=0) for i in range(len(logits_l_sup))]           
            logits_r_sup = [F.interpolate(input=outsup2[0][i], size=targets.shape[2:], mode='bilinear',
                            align_corners=False) for i in range(len(outsup2[0]))]
            logits_r_unsup = [F.interpolate(input=outunsup2[0][i], size=targets.shape[2:], mode='bilinear',
                            align_corners=False) for i in range(len(outunsup2[0]))]
            logits_r = [torch.cat((logits_l_sup[i], logits_l_unsup[i]), dim=0) for i in range(len(logits_l_sup))]
            cps1_logits = [loss_dice_fn(logits_l[i], max_r) for i in range(len(logits_l))]
            cps2_logits = [loss_dice_fn(logits_r[i], max_l) for i in range(len(logits_r))]
            cps1_logits = torch.stack(cps1_logits, dim=0).mean()
            cps2_logits = torch.stack(cps2_logits, dim=0).mean()
            cps_loss = cps1_ensemble + cps2_ensemble + cps1_logits + cps2_logits
            cps_loss *= lamdba_scaling(epoch_num)

            # Supervision loss
            # ensemble
            loss_dice_ensemble = loss_dice_fn(outputs_sup, targets.long())
            loss_dice2_ensemble = loss_dice_fn(outputs_sup2, targets.long())
            # learners
            sup_logits = [loss_dice_fn(logits_l_sup[i], targets.long()) for i in range(len(logits_l_sup))]
            loss_dice_logits = torch.stack(sup_logits, dim=0).sum() / len(sup_logits)
            sup_logits2 = [loss_dice_fn(logits_r_sup[i], targets.long()) for i in range(len(logits_r_sup))]
            loss_dice2_logits = torch.stack(sup_logits2, dim=0).sum() / len(sup_logits2)
            loss_dice_total = loss_dice_ensemble + loss_dice2_ensemble + loss_dice_logits + loss_dice2_logits
            loss = cps_loss + loss_dice_total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)

            optimizer.step()
            optimizer2.step()

            cps1_ensemble_.append(cps1_ensemble.item())
            cps1_logits_.append(cps1_logits.item())
            cps2_ensemble_.append(cps2_ensemble.item())
            cps2_logits_.append(cps2_logits.item())
            losses_train.append(loss.item())
            dices_train_ensemble.append(loss_dice_ensemble.item())
            dices_train_logits.append(loss_dice_logits.item())
            dices_train2_ensemble.append(loss_dice2_ensemble.item())            
            dices_train2_logits.append(loss_dice2_logits.item())

            # update lr
            lr_ = poly_lr_linear_warmup(base_lr, iter, max_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
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
        to_log["train/cps1_ensemble"] = statistics.mean(cps1_ensemble_)
        to_log["train/cps2_ensemble"] = statistics.mean(cps2_ensemble_)
        to_log["train/cps1_logits"] = statistics.mean(cps1_logits_)
        to_log["train/cps2_logits"] = statistics.mean(cps2_logits_)
        to_log["train/sup1_ensemble"] = statistics.mean(dices_train_ensemble)
        to_log["train/sup1_logits"] = statistics.mean(dices_train_logits)
        to_log["train/sup2_ensemble"] = statistics.mean(dices_train2_ensemble)
        to_log["train/sup2_logits"] = statistics.mean(dices_train2_logits)
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
