import os
import numpy as np
import wandb
import torch
from dset import SegmentationDataset
from utils import *
import torch.nn.functional as F

import statistics

from PIL import Image

device = torch.device("cpu")

HPP_DEFAULT = Dotdict(dict(
    best_model_dice=True,  # if True, "best" model is best dice, else best val loss
    run_id="wg360vfw",
    use_train_norm=True
))


def from_txt_files(img_file_path, gt_file_path, test_images_path, gt_test_images_path, config, nuclei_string):
    with open(img_file_path, "r") as f:
        img_files = f.read().splitlines()
    with open(gt_file_path, "r") as f:
        gt_files = f.read().splitlines()
    print("{} test files".format(len(img_files)))

    # get mean and std from test set
    full_path_img = [os.path.join(test_images_path, img_fl) for img_fl in img_files]
    full_path_gt = [os.path.join(test_images_path, gt_fl) for gt_fl in gt_files]

    dices_nearest = []
    dices_area = []
    dices_bilinear = []
    dices_bicubic = []
    gt_pixels = 0
    nearest_pixels = 0
    area_pixels = 0
    bilinear_pixels = 0
    bicubic_pixels = 0
    common_nearest_pixels = 0
    common_area_pixels = 0
    common_bilinear_pixels = 0
    common_bicubic_pixels = 0

    for i, img_file in enumerate(img_files):
        to_log = {}
        gt_path = os.path.join(gt_test_images_path, gt_files[i])
        gt = Image.open(gt_path)
        gt = torch.as_tensor(np.array(gt), dtype=torch.int64)
        gt = torch.where(gt > 0.5, 1, 0).unsqueeze(0).unsqueeze(0).float()
        # only compute dice where there is structure in the data
        if gt.max() > 0:
            gt = gt.to(device)
            mask_list = get_mask_list(gt)

            nearest_pixels += mask_list[0].sum().item()
            area_pixels += mask_list[1].sum().item()
            bilinear_pixels += mask_list[2].sum().item()
            bicubic_pixels += mask_list[3].sum().item()
            gt_pixels += gt.sum().item()

            common_nearest_pixels += (mask_list[0] * gt).sum().item()
            common_area_pixels += (mask_list[1] * gt).sum().item()
            common_bilinear_pixels += (mask_list[2] * gt).sum().item()
            common_bicubic_pixels += (mask_list[3] * gt).sum().item()

            dice_nearest = dice_coeff(gt, mask_list[0])
            dice_area = dice_coeff(gt, mask_list[1])
            dice_bilinear = dice_coeff(gt, mask_list[2])
            dice_bicubic = dice_coeff(gt, mask_list[3])

            dices_nearest.append(dice_nearest.item())
            dices_area.append(dice_area.item())
            dices_bilinear.append(dice_bilinear.item())
            dices_bicubic.append(dice_bicubic.item())
            to_log["every_dice_nearest"] = dice_nearest.item()
            to_log["every_dice_area"] = dice_area.item()
            to_log["every_dice_bilinear"] = dice_bilinear.item()
            to_log["every_dice_bicubic"] = dice_bicubic.item()
            if i % 100 == 0:
                running_dice_nearest = statistics.mean(dices_nearest)
                running_dice_area = statistics.mean(dices_area)
                running_dice_bilinear = statistics.mean(dices_bilinear)
                running_dice_bicubic = statistics.mean(dices_bicubic)
                to_log["running_dice_nearest"] = running_dice_nearest
                to_log["running_dice_area"] = running_dice_area
                to_log["running_dice_bilinear"] = running_dice_bilinear
                to_log["running_dice_bicubic"] = running_dice_bicubic
                to_log["running_3d_dice_nearest"] = 2 * common_nearest_pixels / (gt_pixels + nearest_pixels)
                to_log["running_3d_dice_area"] = 2 * common_area_pixels / (gt_pixels + area_pixels)
                to_log["running_3d_dice_bilinear"] = 2 * common_bilinear_pixels / (gt_pixels + bilinear_pixels)
                to_log["running_3d_dice_bicubic"] = 2 * common_bicubic_pixels / (gt_pixels + bicubic_pixels)

        wandb.log(to_log)

    # get final mean
    avg_dice_nearest = statistics.mean(dices_nearest)
    avg_dice_area = statistics.mean(dices_area)
    avg_dice_bilinear = statistics.mean(dices_bilinear)
    avg_dice_bicubic = statistics.mean(dices_bicubic)
    dim_dice_nearest = 2 * common_nearest_pixels / (gt_pixels + nearest_pixels)
    dim_dice_area = 2 * common_area_pixels / (gt_pixels + area_pixels)
    dim_dice_bilinear = 2 * common_bilinear_pixels / (gt_pixels + bilinear_pixels)
    dim_dice_bicubic = 2 * common_bicubic_pixels / (gt_pixels + bicubic_pixels)

    wandb.log({"dice_nearest": avg_dice_nearest, "dice_area": avg_dice_area, "dice_bilinear": avg_dice_bilinear, "dice_bicubic": avg_dice_bicubic,
               "3D_dice_nearest": dim_dice_nearest, "3D_dice_area": dim_dice_area, "3D_dice_bilinear": dim_dice_bilinear,
               "3D_dice_bicubic": dim_dice_bicubic})


def fold(unfolded_tensor, H, W, kernel_size, stride):
    results = unfolded_tensor.permute(1, 0, 2, 3)  # [nclasses, nb_patches_all, kernel_size, kernel_size]
    patches = results.contiguous().view(1, 1, -1,
                                        kernel_size * kernel_size)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches = patches.permute(0, 1, 3, 2)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches = patches.contiguous().view(patches.size(0), 1 * kernel_size * kernel_size,
                                        -1)  # [B, C*prod(kernel_size), L] as expected by Fold
    output = F.fold(patches, output_size=(H, W),
                    kernel_size=kernel_size, stride=stride)
    # Fold sums overlapping patches, so we need to divide each pixel by the number of patches it was in
    counter = F.fold(torch.ones_like(patches), output_size=(H, W), kernel_size=kernel_size, stride=stride)
    averaged_output = output / counter

    return averaged_output


def get_mask_list(x):
    # dim_crop is kernel size
    kernel_size = 2048
    stride = kernel_size // 4
    # pad
    pad_h = stride - (x.size(2) - kernel_size) % stride
    pad_w = stride - (x.size(3) - kernel_size) % stride
    x_padded = F.pad(x, (pad_w // 2,
                         pad_w - pad_w // 2,
                         pad_h // 2,
                         pad_h - pad_h // 2),
                     mode="reflect")
    H, W = x_padded.size(2), x_padded.size(3)
    # unfold
    patches = x_padded.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.contiguous().view(patches.size(0), -1, kernel_size,
                                        kernel_size)  # [1, nb_patches_all, config.dim_crop, config.dim_crop]
    patches = patches.permute(1, 0, 2, 3).contiguous()  # [nb_patches_all, 1, kernel_size, kernel_size]
    # do all interpolation methods
    patches_nearest_d = torch.nn.functional.interpolate(input=patches.clone(), size=512, mode="nearest")
    patches_area_d = torch.nn.functional.interpolate(input=patches.clone(), size=512, mode="area")
    patches_bilinear_d = torch.nn.functional.interpolate(input=patches.clone(), size=512, mode="bilinear")
    patches_bicubic_d = torch.nn.functional.interpolate(input=patches.clone(), size=512, mode="bicubic")

    patches_nearest = torch.nn.functional.interpolate(input=patches_nearest_d, size=kernel_size, mode="nearest")
    patches_area = torch.nn.functional.interpolate(input=patches_area_d, size=kernel_size, mode="area")
    patches_bilinear = torch.nn.functional.interpolate(input=patches_bilinear_d, size=kernel_size, mode="bilinear")
    patches_bicubic = torch.nn.functional.interpolate(input=patches_bicubic_d, size=kernel_size, mode="bicubic")
    # fold back together
    nearest = fold(patches_nearest, H, W, kernel_size, stride)
    area = fold(patches_area, H, W, kernel_size, stride)
    bilinear = fold(patches_bilinear, H, W, kernel_size, stride)
    bicubic = fold(patches_bicubic, H, W, kernel_size, stride)

    # Crop back to original size
    final_nearest = nearest[:, :, pad_h // 2:-(pad_h - pad_h // 2), pad_w // 2:-(pad_w - pad_w // 2)]
    final_area = area[:, :, pad_h // 2:-(pad_h - pad_h // 2), pad_w // 2:-(pad_w - pad_w // 2)]
    final_bilinear = bilinear[:, :, pad_h // 2:-(pad_h - pad_h // 2), pad_w // 2:-(pad_w - pad_w // 2)]
    final_bicubic = bicubic[:, :, pad_h // 2:-(pad_h - pad_h // 2), pad_w // 2:-(pad_w - pad_w // 2)]

    mask_nearest = torch.where(final_nearest > 0.5, 1, 0)
    mask_area = torch.where(final_area > 0.5, 1, 0)
    mask_bilinear = torch.where(final_bilinear > 0.5, 1, 0)
    mask_bicubic = torch.where(final_bicubic > 0.5, 1, 0)
    return [mask_nearest, mask_area, mask_bilinear, mask_bicubic]


eval_run = wandb.init(project="eval_interpolation", entity="ohsu-cv", config=HPP_DEFAULT,
                      settings=wandb.Settings(start_method='fork'))
api = wandb.Api()
train_run = api.run("ohsu-cv/ceecnet/{}".format(HPP_DEFAULT.run_id))
config = Dotdict(train_run.config)

txt_file_path = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/new_10/' + config.e_name + '/' + config.data_num + '/'

if config.nuclei:
    nuclei_string = "nuclei"
    img_txt = os.path.join(txt_file_path, config.e_name + '_test_images.txt')
    gt_img_txt = os.path.join(txt_file_path, config.e_name + '_test_nucleus.txt')
else:
    nuclei_string = "nucleoli"
    if config.e_name == "PPT":
        img_txt = os.path.join(txt_file_path, config.e_name + '_test_images.txt')
    else:
        img_txt = os.path.join(txt_file_path, config.e_name + '_test_images_nucleolus.txt')
    gt_img_txt = os.path.join(txt_file_path, config.e_name + '_test_nucleolus.txt')

if config.e_name == "101a":
    test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/' + config.e_name + '/images_new/'
    gt_test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/EM/loftis/loftis/data/101a/' + nuclei_string + '-labels/'
elif config.e_name == "101b":
    test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/EM/EM_Data/Confidential-RestrictedAccess/Data/16113-101/101b-1/Helios/Segmentation/Full Dataset/Images/Images_as_Tiffs/'
    nuclei_string_caps = nuclei_string.capitalize()
    gt_test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/EM/EM_Data/Confidential-RestrictedAccess/Data/16113-101/101b-1/Helios/Segmentation/Full Dataset/' + nuclei_string_caps + '/' + nuclei_string_caps + ' Tiffs/'
elif config.e_name == "PPT":
    test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/' + config.e_name + '/images_new/'
    gt_test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/EM/loftis/loftis/data/4373T1/' + nuclei_string + '-labels/'
    if not config.nuclei:
        gt_test_images_path += "Merge/"

from_txt_files(img_txt, gt_img_txt, test_images_path, gt_test_images_path, config, nuclei_string)
