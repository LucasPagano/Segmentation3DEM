import sys


sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
import os
import statistics
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PyTorch_Models.DTC.code.networks.unet import SmallUNet
from PyTorch_Models.DTC.code.networks.senformer import SenFormer

from PyTorch_Models.CEECNET.dset import SegmentationDataset
from PyTorch_Models.CEECNET.ceecnet import XNetSegmentation
from PyTorch_Models.CEECNET.ceecnet2 import XNetSegmentation as XNetSegmentation2
from PyTorch_Models.CEECNET.utils import *

HPP_DEFAULT = Dotdict(dict(
    best_model_dice=True,  # if True, "best" model is best dice, else best val loss
    run_id="2ahohm4s",
    use_train_norm=True
))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def from_txt_files(img_file_path, gt_file_path, test_images_path, gt_test_images_path, config, model, nuclei_string):
    with open(img_file_path, "r") as f:
        img_files = f.read().splitlines()
    with open(gt_file_path, "r") as f:
        gt_files = f.read().splitlines()
    print("{} test files".format(len(img_files)))

    # get mean and std from test set
    full_path_img = [os.path.join(test_images_path, img_fl) for img_fl in img_files]
    full_path_gt = [os.path.join(test_images_path, gt_fl) for gt_fl in gt_files]
    print(len(full_path_img))
    print(len(full_path_gt))
    dataset = SegmentationDataset(config, "", "", False, True, full_path_img, full_path_gt, HPP_DEFAULT.use_train_norm)

    classes_out = ["__background__", nuclei_string]
    dices = []
    gt_pixels = 0
    pred_pixels = 0
    common_pixels = 0

    empty_images = 0
    bad_images = 0
    for i, img_file in enumerate(img_files):
        to_log = {}
        img_path = os.path.join(test_images_path, img_file)
        gt_path = os.path.join(gt_test_images_path, gt_files[i])
        img = Image.open(img_path)
        img = TF.to_tensor(img).unsqueeze(0)
        if config.normalize:
            mean, std = dataset.get_norm_values()
            img = TF.normalize(img, mean=mean, std=std)
        gt = Image.open(gt_path)
        gt = torch.as_tensor(np.array(gt), dtype=torch.int64)
        gt = torch.where(gt > 0.5, 1, 0).byte()
        # only compute dice where there is structure in the data
        if gt.max() > 0:
            mask_pred = sliding_window(img, model, config)
            gt = gt.to(device)
            pred_pixels += mask_pred.sum().item()
            gt_pixels += gt.sum().item()
            common_pixels += (mask_pred * gt).sum().item()
            dice = dice_coeff(gt, mask_pred)
            dices.append(dice.item())
            to_log["every_dice"] = dice.item()
            # log every 100 images
            if i % 100 == 0:
                log_mask = detach_to_numpy(mask_pred.squeeze()).astype(np.uint8)
                img2 = detach_to_numpy(img.squeeze())
                gt2 = detach_to_numpy(gt.squeeze()).astype(np.uint8)
                to_log_img = wb_mask(bg_img=img2, pred_mask=log_mask, true_mask=gt2, classes_out=classes_out)
                running_dice = statistics.mean(dices)
                to_log["img_{}".format(i)] = to_log_img
                to_log["running_dice"] = running_dice
                to_log["running_3d_dice"] = 2 * common_pixels / (gt_pixels + pred_pixels)

            # log images where model did poorly
            if dice.item() < 0.3:
                bad_images += 1
                log_mask = detach_to_numpy(mask_pred.squeeze()).astype(np.uint8)
                img2 = detach_to_numpy(img.squeeze())
                gt2 = detach_to_numpy(gt.squeeze()).astype(np.uint8)
                to_log_img = wb_mask(bg_img=img2, pred_mask=log_mask, true_mask=gt2, classes_out=classes_out)
                to_log["bad_img/{}_{}".format(i, bad_images)] = to_log_img

        else:
            empty_images += 1
            to_log["every_dice"] = -1

        wandb.log(to_log)

    avg_dice = statistics.mean(dices)
    dim_dice = 2 * common_pixels / (gt_pixels + pred_pixels)
    wandb.log({"dice": avg_dice, "3D_dice": dim_dice, "empty_images": empty_images})


def sliding_window(x, model, config):
    # dim_crop is kernel size
    if config.downsize:
        kernel_size = config.dim_crop
        stride = config.dim_crop // 4
    else:
        kernel_size = config.dimensions_input
        stride = config.dimensions_input // 4

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
    patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)  # [1, nb_patches_all, config.dim_crop, config.dim_crop]
    patches = patches.permute(1, 0, 2, 3).contiguous()  # [nb_patches_all, 1, kernel_size, kernel_size]
    # downsize if done during training
    if config.downsize:
        patches = torch.nn.functional.interpolate(input=patches,
                                                  size=config.dimensions_input,
                                                  mode="bilinear")  # [nb_patches_all, 1, dimensions_input, dimensions_input]

    # apply model to every patch
    batch_size = config.batch_size
    if torch.cuda.device_count() > 1:
        batch_size = batch_size * torch.cuda.device_count()
    outs = []
    for i in range(0, patches.size(0), batch_size):
        ins = patches[i:i + batch_size].to(device)
        if config.model_type_eval == "ceecnet":
            # we only care about the mask prediction outputs
            outs.append(detach_to_numpy(model(ins)[0]))
        elif config.model_type_eval == "model_smp":
            outs.append(detach_to_numpy(model(ins)))
        elif config.model_type_eval == "smallunet":
            outs.append(detach_to_numpy(model(ins)))
        else:
            outs.append(detach_to_numpy(model.forward_test(ins)))

    segmentation_masks = torch.tensor(np.concatenate(outs, axis=0)).to(device)  # [nb_patches_all, nbclasses, dimensions_input, dimensions_input]

    # Upsample if we downsized
    if config.downsize:
        segmentation_masks = torch.nn.functional.interpolate(input=segmentation_masks, size=kernel_size, mode="bilinear")

    # fold back together
    results = segmentation_masks.permute(1, 0, 2, 3)  # [nclasses, nb_patches_all, kernel_size, kernel_size]
    patches = results.contiguous().view(1, config.nclasses, -1, kernel_size * kernel_size)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches = patches.permute(0, 1, 3, 2)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches = patches.contiguous().view(patches.size(0), config.nclasses * kernel_size * kernel_size,
                                        -1)  # [B, C*prod(kernel_size), L] as expected by Fold
    output = F.fold(patches, output_size=(H, W),
                    kernel_size=kernel_size, stride=stride)
    # Fold sums overlapping patches, so we need to divide each pixel by the number of patches it was in
    counter = F.fold(torch.ones_like(patches), output_size=(H, W), kernel_size=kernel_size, stride=stride)
    averaged_output = output / counter

    # Crop back to original size
    final = averaged_output[:, :, pad_h // 2:-(pad_h - pad_h // 2), pad_w // 2:-(pad_w - pad_w // 2)]

    # get predicted mask
    if config.nclasses > 1:
        mask = torch.argmax(final, dim=1)
    else:
        mask = torch.where(final > 0.5, 1, 0)

    return mask


def run_eval(run_id=None):
    if run_id is not None:
        HPP_DEFAULT.run_id = run_id
    eval_run = wandb.init(project="eval_nucl", entity="ohsu-cv", config=HPP_DEFAULT, settings=wandb.Settings(start_method='fork'))
    api = wandb.Api()
    try:
        train_run = api.run("ohsu-cv/ceecnet/{}".format(HPP_DEFAULT.run_id))
    except Exception:
        train_run = api.run("ohsu-cv/DTC/{}".format(HPP_DEFAULT.run_id))
        os.chdir("../DTC/code")
    config = Dotdict(train_run.config)
    print(config)
    # remember what we're evaluating
    wandb.config.update(config)
    eval_run.tags = train_run.tags + [train_run.name]
    file_name = "model_best_dice.pt" if HPP_DEFAULT.best_model_dice else "model_best_val.pt"
    checkpoint = torch.load("./models/{}/{}".format(HPP_DEFAULT.run_id, file_name), map_location=device)
    print(checkpoint.keys())
    print("Best epoch : {}".format(checkpoint["epoch"]))
    if config.get("ceecnet"):
        config.model_type_eval = "ceecnet"
        try:
            model = XNetSegmentation(config).to(device)
        except Exception:
            model = XNetSegmentation2(config).to(device)
    elif config.get("model_smp") is not None:
        config.model_type_eval = "model_smp"
        model = get_smp_model(config).to(device)
    elif config.get("cps_weight") is not None and "Senformer" not in eval_run.tags:
        model = SmallUNet(1, 1, config.start_channels).to(device)
        config.model_type_eval = "smallunet"
    else:
        model = SenFormer(num_heads=8, branch_depth=6, in_chans=1, num_classes=config.nclasses, mlp_ratio=4.,
                          qkv_bias=True,
                          qk_scale=None, drop=0.,
                          attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1.e-15,
                          align_corners=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval()

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
        test_images_path = "/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/PPT/images_new/"
        gt_test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/EM/loftis/loftis/data/4373T1/' + nuclei_string + '-labels/'
        if not config.nuclei:
            gt_test_images_path += "Merge/"

    with torch.no_grad():
        from_txt_files(img_txt, gt_img_txt, test_images_path, gt_test_images_path, config, model, nuclei_string)


if __name__ == "__main__":
    run_eval()
