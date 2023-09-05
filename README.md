# Segmentation 3D EM

This repository contains all the code necessary to train and evaluate a variety of fully supervised models (UNet, Unet++, Linknet, FPN, FractalResNet, CEECNet, ...) as well as the semi-supervised Cross Pseudo supervision framework on the task of semantic segmentation as described in [citation].
All trainings and evaluations were conducted on a V100 Tesla GPU.

### Setup
#### 🔨 Environment
Clone this repository and enter it:  `git clone https://github.com/LucasPagano/Segmentation3DEM.git && cd Segmentation3DEM`.\
Create a conda environment `conda create -n seg3D python=3.8`, and activate it `conda activate seg3D`. \
Install the requirements `pip install -r requirements.txt`. \
Here is a full script for setting up a conda environment to use Segmentation3DEM:\
```sh
git clone https://github.com/LucasPagano/Segmentation3DEM.git && cd Segmentation3DEM 
conda create -n seg3D python=3.8
conda activate seg3D
pip install -r requirements.txt
```

#### Data
Download your images and masks and split them into training and validation folders (so you end up with 4 different folders). Add these folders in the HPP_DEFAULT dictionary of the training file you use at `train_data_path, train_masks_path, val_data_path, val_masks_path`. \
The files in the folders will be listed and files with a mask matched together following a naming convention. Naming convention assumed is that the label number is just before the extension, and that other numbers in the file name are separated from the label number eg: \ 
A45.60Tile3.tif / Label_A45.60Tile_Nuclei3.tif: label 3, \
C089.1.Big_Nucl004.png / 101bNuclei_labels_004.tif : label 004, ...

To train fully supervised models, make your hyper-parameter choices and fill in the location of your images and masks folders in the PyTorch_Models/CEECNET/train.py HPP_DEFAULT dictionary. Then run sh run_train.sh in a bash shell with a slurm environment.

To train semi-supervised models, make your hyper-parameter choice and fill in the location of your images and masks folders in the PyTorch_Models/CPS/code/[file].py HPP_DEFAULT dictionary corresponding to the technique you want to use (e.g. use train.py for a training close to the original CPS paper, use cross_pseudo_supervision[_cutmix].py for our implementation with the Soft Dice loss. We included the training files of our tests of CPS with Senformer (Bousselham et al. [2021](https://doi.org/10.48550/arXiv.2111.13280)) but we did not include results in the paper as they performed lower than average.

### Train your own models with the Cross Pseudo Supervision framework
If you want to train your own Pytorch model with the semi-supervised framework, you can simply add the model in a .py file and change 
``` 
model = get_smp_model(HPP_DEFAULT).to(device)
model2 = get_smp_model(HPP_DEFAULT).to(device)
```
to
``` 
model = YourModel.to(device)
model2 = YourModel.to(device)
```

### Citation

If you find this repository useful, please consider citing our work:

```

```