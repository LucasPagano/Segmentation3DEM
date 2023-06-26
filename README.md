# Segmentation 3D EM

This repository contains all the code necessary to train and evaluate a variety of fully supervised models (UNet, Unet++, Linknet, FPN, FractalResNet, CEECNet, ...) as well as the semi-supervised Cross Pseudo supervision framework on the task of semantic segmentation as described in [citation].
All trainings and evaluations were conducted on a V100 Tesla GPU.

## Setup
Download your images and masks and split them into training and validation folders (so you end up with 4 different folders).

To train fully supervised models, make your hyper-parameter choices and fill in the location of your images and masks folders in the PyTorch_Models/CEECNET/train.py HPP_DEFAULT dictionary. Then run sh run_train.sh in a bash shell with a slurm environment.

To train semi-supervised models, make your hyper-parameter choice and fill in the location of your images and masks folders in the PyTorch_Models/CPS/code/train.py HPP_DEFAULT dictionary corresponding to the technique you want to use (e.g. use train.py for a training close to the original CPS paper, use cross_pseudo_supervision[_cutmix].py for our implementation with the Soft Dice loss. We included the training files of our tests of CPS with Senformer (Bousselham et al. [2021](https://doi.org/10.48550/arXiv.2111.13280) but we did not include results in the paper as they performed lower than average.)