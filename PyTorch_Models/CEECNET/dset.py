import os
import statistics
from os import listdir
from os.path import isfile, join
from pathlib import Path
import time
import cv2
import skimage.exposure as exposure
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PyTorch_Models.CEECNET.utils import get_boundary, get_distance, detach_to_numpy, Dotdict


class SegmentationDataset(Dataset):
    """

    """
    def __init__(self, config, image_dir, target_dir, train=True, eval_mode=False, img_paths=None, gt_paths=None, use_train_norm=None, pad_mask=False):
        if eval_mode and img_paths is not None and gt_paths is not None:
            if use_train_norm:
                img_string = "images" if config.e_name == "PPT" or config.nuclei else "nucleolus_images"
                msk_string = "nucleus_mask" if config.nuclei else "nucleolus_mask"
                image_dir = "/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/new_10/" + config.e_name + "/25_9/" + img_string + "/1"
                target_dir = "/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/new_10/" + config.e_name + "/25_9/" + msk_string + "/1"
                self.image_paths = sorted([join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))])
                self.target_paths = sorted([join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))])
            else:
                self.image_paths = img_paths
                self.target_paths = gt_paths
        else:
            self.image_paths = sorted([join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))])
            self.target_paths = sorted([join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))])
        self.config = config
        self.pad_mask = pad_mask
        self.train = train  # True for train, False for val
        if self.config.normalize:
            self.norm_mean = None
            self.norm_std = None
            self.normalize()

    def normalize(self):
        print("Computing normalize stats")
        means = []
        stds = []
        for img_path in self.image_paths:
            img = TF.to_tensor(Image.open(img_path))
            means.append(img.mean().item())
            stds.append(img.std().item())
        self.norm_mean = statistics.mean(means)
        self.norm_std = statistics.mean(stds)

    def transform(self, image, mask):
        if self.config.downsize:
            # if image not big enough, pad before cropping
            w, h = image.size
            pad_w = self.config.dim_crop - w if self.config.dim_crop > w else 0
            pad_h = self.config.dim_crop - h if self.config.dim_crop > h else 0
            if pad_w != 0 or pad_h != 0:
                image = TF.pad(image, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], padding_mode="reflect")
                if self.pad_mask:
                    mask = TF.pad(mask, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=255)
                else:
                    mask = TF.pad(mask, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], padding_mode="reflect")

            found = False
            while not found:

                # Random crop
                # [H,W] -> [dim_crop,dim_crop]
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.config.dim_crop, self.config.dim_crop))

                img = TF.crop(image, i, j, h, w)
                msk = TF.crop(mask, i, j, h, w)

                # if image is only background, keep it with probability set in HPP, else keep it
                if msk.getextrema()[1] > 0 or np.random.uniform() < self.config.keep_empty_output_prob:
                    found = True
                    image = img
                    mask = msk

            image = TF.to_tensor(image).unsqueeze(0)

            # Downsize
            # [dim_crop,dim_crop] -> [1,dimensions_input,dimensions_input]
            # interpolate expects inputs of size [B, channels, H W]
            image = F.interpolate(input=image, size=self.config.dimensions_input, mode="bilinear").squeeze(0)

            mask = F.interpolate(
                    input=torch.as_tensor(np.array(mask), dtype=torch.int64).unsqueeze(0).unsqueeze(0).float(),
                    size=self.config.dimensions_input, mode="bilinear").squeeze(0)

        else:
            found = False
            while not found:
                # Random crop
                # [H,W] -> [dim_input,dim_input]
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(self.config.dimensions_input, self.config.dimensions_input))
                img = TF.to_tensor(np.array(TF.crop(image, i, j, h, w), dtype=np.uint8))
                msk = torch.as_tensor(np.array(TF.crop(mask, i, j, h, w)), dtype=torch.int64).unsqueeze(0)
                # if image is only background, keep it with probability set in HPP, else keep it
                if msk.max() > 0 or np.random.uniform() < self.config.keep_empty_output_prob:
                    found = True
                    image = img
                    mask = msk

        if self.train:
            # random rescale
            if self.config.random_rescale:
                if torch.rand(1) > 0.5:
                    random_factor = np.random.uniform(low=0.75, high=1.25)
                    matrix_transform = cv2.getRotationMatrix2D((0, 0), 0, random_factor)
                    image = torch.as_tensor(cv2.warpAffine(detach_to_numpy(image.permute(1, 2, 0)), matrix_transform,
                                                           tuple(image.size()[1:]), flags=cv2.INTER_AREA,
                                                           borderMode=cv2.BORDER_REFLECT_101)).unsqueeze(0)
                    mask = torch.as_tensor(cv2.warpAffine(detach_to_numpy(mask.permute(1, 2, 0)).astype(np.uint8), matrix_transform,
                                                          tuple(mask.size()[1:]), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)).unsqueeze(
                        0)

        if self.config.normalize:
            image = TF.normalize(image, mean=self.norm_mean, std=self.norm_std)

        # Binarize mask and cast to uint8
        if self.pad_mask:
            # padding
            mask[mask >= 50] = 255
            # classes
            mask[(mask > 0.5) & (mask < 50)] = 1
            mask[mask < 0.5] = 0
            mask = mask.byte()
        else:
            mask = torch.where(mask > 0.5, 1, 0).byte()

        if self.train:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)

            # Random vertical flip
            if torch.rand(1) > 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)

        # make mask one hot encoding for loss
        # dim 0 is background class, dim 1 is target class
        # mask = torch.stack((1 - mask, mask)).squeeze()

        # get boundary and distance map
        if self.config.get("ceecnet", True):
            boundary = get_boundary(mask)
            distance_map = get_distance(mask)
            targets = [mask, boundary, distance_map]
        else:
            targets = [mask]

        return image, targets

    def __getitem__(self, index):
        index_nb_crop = index // self.config.nb_crop_per_image
        image = Image.open(self.image_paths[index_nb_crop])
        mask = Image.open(self.target_paths[index_nb_crop])

        x, y = self.transform(image, mask)

        return x, y

    def __len__(self):
        return len(self.image_paths) * self.config.nb_crop_per_image

    def get_norm_values(self):
        return self.norm_mean, self.norm_std

    def norm(self, image):
        TF.normalize(image, self.norm_mean, self.norm_std)

    def inverse_norm(self, image):
        return TF.normalize(image, (-self.norm_mean / self.norm_std), (1.0 / self.norm_std))


class UnsupervisedSegmentationDataset(SegmentationDataset):
    def __init__(self, config, image_dir, target_dir, train=True, eval_mode=False, img_paths=None, gt_paths=None, use_train_norm=None):
        super().__init__(config, image_dir, target_dir, train, eval_mode, img_paths, gt_paths, use_train_norm)

        if config.e_name == "101a":
            test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/' + config.e_name + '/images_new/'
        elif config.e_name == "101b":
            test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/EM/EM_Data/Confidential-RestrictedAccess/Data/16113-101/101b-1/Helios/Segmentation/Full Dataset/Images/Images_as_Tiffs/'
        elif config.e_name == "PPT":
            test_images_path = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM/nuclei_new/data/' + config.e_name + '/images_new/'
        else:
            raise Exception

        unsupervised_images_dir = test_images_path
        self.unsupervised_image_paths = sorted(
            [join(unsupervised_images_dir, f) for f in listdir(unsupervised_images_dir) if isfile(join(unsupervised_images_dir, f))])
        self.unsupervised_counter = 0

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        unsupervised_image = None
        # some of the files aren't images
        while unsupervised_image is None:
            try:
                unsupervised_image = Image.open(self.unsupervised_image_paths[self.unsupervised_counter])
                unsupervised_image = self.unsupervised_transform(unsupervised_image)
            except Exception:
                self.unsupervised_counter += 1
        self.unsupervised_counter += 1
        if self.unsupervised_counter == len(self.unsupervised_image_paths):
            self.unsupervised_counter = 0
        return image, target, unsupervised_image

    def unsupervised_transform(self, image):
        # simply performs a random crop in the image followed by random rotations and normalization

        # if image not big enough, pad before cropping
        w, h = image.size
        pad_w = self.config.dim_crop - w if self.config.dim_crop > w else 0
        pad_h = self.config.dim_crop - h if self.config.dim_crop > h else 0
        if pad_w != 0 or pad_h != 0:
            image = TF.pad(image, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], padding_mode="reflect")

        if self.config.downsize:
            # Random crop
            # [H,W] -> [dim_crop,dim_crop]
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.config.dim_crop, self.config.dim_crop))

            img = TF.crop(image, i, j, h, w)

            image = TF.to_tensor(img).unsqueeze(0)

            # Downsize
            # [dim_crop,dim_crop] -> [1,dimensions_input,dimensions_input]
            # interpolate expects inputs of size [B, channels, H W]
            image = F.interpolate(input=image, size=self.config.dimensions_input, mode="bilinear").squeeze(0)
        else:
            # Random crop
            # [H,W] -> [dim_input,dim_input]
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.config.dimensions_input, self.config.dimensions_input))
            image = TF.to_tensor(np.array(TF.crop(image, i, j, h, w), dtype=np.uint8))

        if self.config.normalize:
            image = TF.normalize(image, mean=self.norm_mean, std=self.norm_std)

        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)

        # Random vertical flip
        if torch.rand(1) > 0.5:
            image = TF.vflip(image)
        return image

class MultipleUnsupervisedSegmentationDataset(Dataset):
    def __init__(self, *unsuperviseddsets, histo_match=False):
        self.dsets = unsuperviseddsets
        self.path = None
        if histo_match:
            self.histo_match()

    def __getitem__(self, index):
       return tuple(d[index] for d in self.dsets)

    def __len__(self):
        return min(len(d) for d in self.dsets)

    def histo_match(self):
        ## match histograms between image of the labelled dset (dset1, target domain) and unlabelled one (dset, source domain)
        # create tmp directory
        t = str(time.time()).replace(".", "")
        path = "tmp/" + t + "/"
        self.path = path
        path_sup = path + "sup/"
        Path(path_sup).mkdir(parents=True, exist_ok=False)
        # save matched images in tmp dir
        ref_image = np.array(Image.open(self.dsets[0].image_paths[0]))
        for ipath in self.dsets[1].image_paths:
            im = np.array(Image.open(ipath))
            matched = exposure.match_histograms(im, ref_image, multichannel=False)
            m = Image.fromarray(matched)
            spath = path_sup + os.path.basename(ipath).replace(".png", ".tif")
            m.save(spath)

        path_unsup = path + "unsup/"
        Path(path_unsup).mkdir(parents=True, exist_ok=False)
        for ipath in self.dsets[1].unsupervised_image_paths:
            try:
                im = np.array(Image.open(ipath))
                matched = exposure.match_histograms(im, ref_image, multichannel=False)
                m = Image.fromarray(matched)
                upath = path_unsup + os.path.basename(ipath).replace(".png", ".tif")
                m.save(upath)
            except Exception:
                print("e")

        # change dset file list to tmp dir files
        self.dsets[1].image_paths = sorted(
            [join(path_sup, f) for f in listdir(path_sup) if isfile(join(path_sup, f))])
        # change dset file list to tmp dir files
        self.dsets[1].unsupervised_image_paths = sorted(
            [join(path_unsup, f) for f in listdir(path_unsup) if isfile(join(path_unsup, f))])

        # We changed values in images so recompute norm stats
        self.dsets[1].normalize()

    def cleanup(self):
        import shutil
        shutil.rmtree(self.path)
