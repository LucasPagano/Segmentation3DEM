# Segmentation 3D EM

This repository contains all the code necessary to train and evaluate a variety of fully supervised models (UNet, Unet++, Linknet, FPN, FractalResNet, CEECNet, ...) as well as the semi-supervised Cross Pseudo supervision framework on the task of semantic segmentation as described in [citation].
All trainings and evaluations were conducted on a V100 Tesla GPU.

## Setup
Download your images and masks and split them into training and validation folders (so you end up with 4 different folders).

To train fully supervised models, fill in the location of your images and masks folders in the train.py HPP_DEFAULT dictionnary (#MISC section). Then run sh run_train.sh in a bash shell with a slurm environment.

To train semi-supervised models,  fill in the location of your images and masks folders in the train.py HPP_DEFAULT dictionnary (#MISC section).