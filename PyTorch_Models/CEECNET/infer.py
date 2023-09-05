import shutil
import sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
import matplotlib.image
import tifffile as tiff
import os
import statistics
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PyTorch_Models.CPS.code.networks.unet import SmallUNet
from PyTorch_Models.CPS.code.networks.senformer import SenFormer
from collections import defaultdict
from PyTorch_Models.CEECNET.dset import SegmentationDataset
from PyTorch_Models.CEECNET.ceecnet import XNetSegmentation
from PyTorch_Models.CEECNET.ceecnet2 import XNetSegmentation as XNetSegmentation2
from PyTorch_Models.CEECNET.utils import *


HPP_DEFAULT = Dotdict(dict(
    best_model_dice=True,  # if True, "best" model is best dice, else best val loss
    run_id="68uagql8",
    use_train_norm=False,
))

def sliding_window(x, model, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def run_infer(run_id=None, infer_name=None, image_dir=None):
    if run_id is not None:
        HPP_DEFAULT.run_id = run_id
    if infer_name is None:
        infer_name = HPP_DEFAULT.run_id
    eval_dir = os.path.abspath("../infer/{}/".format(infer_name))

    api = wandb.Api()
    try:
        train_run = api.run("ohsu-cv/ceecnet/{}".format(HPP_DEFAULT.run_id))
    except Exception:
        train_run = api.run("ohsu-cv/3DSeg_standalone/{}".format(HPP_DEFAULT.run_id))
        os.chdir("../CPS/code")
    config = Dotdict(train_run.config)
    if image_dir is None:
        image_dir = config.train_data_path
    nuclei_string = "nuclei" if config.nuclei else "nucleoli"
    # remember what we're evaluating
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    elif config.get("cps_weight") is not None and "Senformer" not in train_run.tags:
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
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir)
    with torch.no_grad():
        full_path_img = [os.path.join(image_dir, img_fl) for img_fl in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, img_fl))]
        # get norm values
        dataset = SegmentationDataset(config, image_dir, eval_dir, False)
        dmean, dstd = dataset.get_norm_values()
        for image_path in full_path_img:
            img = Image.open(image_path)
            img = TF.to_tensor(img).unsqueeze(0)
            if config.normalize:
                img = TF.normalize(img, mean=dmean, std=dstd)
            mask_pred = sliding_window(img, model, config)
            log_mask = detach_to_numpy(mask_pred.squeeze()).astype(np.uint8)
            log_mask_path = image_path.split("/")[-1].replace(".", "_mask.")
            tiff.imsave(os.path.join(eval_dir, log_mask_path), log_mask)

if __name__ == "__main__":
    run_infer()
