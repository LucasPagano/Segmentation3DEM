import numpy
import os
import sys

from PIL import Image, ImageDraw
from PIL.ImageOps import autocontrast as dynexp

sys.path.insert(0, '/Users/firetiti/NN/FiReTiTiPyLib/')
import ImageDataGenerator
import ImagesIO
import Normalizers

sys.path.insert(0, '/Users/firetiti/NN/FiReTiTiPyLib/FiReTiTiPyTorchLib/')
import FiReTiTiPyTorchLib_Datasets

import torch
from torch.utils.data import DataLoader

import torchvision
sys.path.insert(0, '/Users/firetiti/Downloads/DL/TorchVision/')
import TorchVision.detection.utils as utils





InputsNormalizers  = None #Normalizers.Basic(MaxValue=65535.0)
#InputsNormalizers  = [Normalizers.Basic(MaxValue=65535.0)]
OutputsNormalizers = None



Dimensions = 256
Shuffle = False

Path = "/Users/firetiti/NN/DataSets/Test Color/"




Generator = ImageDataGenerator.Generator(ChannelFirst=True)
Generator.setShuffle(True)
Generator.setFlip(False)
Generator.setRotate90x(False)
Generator.setMaxShiftRange(10000)
Generator.setCropPerImage(2)
Generator.setKeepEmptyOutputProbability(1.0)
#Generator.LoadInputs("/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/002 - Cropped/Originals Stretched/", OnTheFly=False, Classification=False)
Generator.LoadInputs("/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/002 - Cropped - Small/Originals Stretched/", OnTheFly=False, Classification=False)
Generator.LoadOutputs("/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/002 - Cropped - Small/Labels/")
Generator.setInputsDimensions(Dimensions, Dimensions)
Generator.setOutputsDimensions(Dimensions, Dimensions)
#Dataset = Generator.PyTorchDataset(1, InputsNormalizers=InputsNormalizers, OutputsNormalizers=None)
gen = Generator.PyTorch(5, InputsNormalizers=InputsNormalizers, OutputsNormalizers=None, Workers=1)
print("Dataset created.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warning = "" if str(device) == "cuda" else "WARNING - "
print(warning + "Device = " + str(device))

nbEpochs = 1
for epoch in range(nbEpochs):
	for b, batch in enumerate(gen):
		#X, Y = batch["input"].to(device, dtype=torch.float), batch["output"].to(device, dtype=torch.float)
		X, Y = batch["input"], batch["output"]
		#print(len(X))
		#print(X.shape)
print("All done")
sys.exit(0)

"""
Datasets = []

inputs  = Path + "/In 1/"
outputs = Path + "/GT 1/"

Generator1 = ImageDataGenerator.Generator(ChannelFirst=True)
Generator1.setShuffle(Shuffle)
Generator1.setCropPerImage(1)
Generator1.setKeepEmptyOutputProbability(1.0)
Generator1.LoadInputs(inputs, OnTheFly=False, Classification=False)
Generator1.LoadOutputs(outputs)
Generator1.setInputsDimensions(Dimensions, Dimensions)
Generator1.setOutputsDimensions(Dimensions, Dimensions)
Dataset1 = Generator1.PyTorchDataset(3, InputsNormalizer=InputsNormalizer, OutputsNormalizers=OutputsNormalizers)
Datasets.append({"Dataset":Dataset1, "Length":int(Dataset1.__len__()/2)})
print("Dataset 1 created.")



inputs  = Path + "/In 2/"
outputs = Path + "/GT 2/"

Generator2 = ImageDataGenerator.Generator(ChannelFirst=True)
Generator2.setShuffle(Shuffle)
Generator2.setCropPerImage(1)
Generator2.setKeepEmptyOutputProbability(1.0)
Generator2.LoadInputs(inputs, OnTheFly=False, Classification=False)
Generator2.LoadOutputs(outputs)
Generator2.setInputsDimensions(Dimensions, Dimensions)
Generator2.setOutputsDimensions(Dimensions, Dimensions)
Dataset2 = Generator2.PyTorchDataset(3, InputsNormalizer=InputsNormalizer, OutputsNormalizers=OutputsNormalizers)
Datasets.append({"Dataset":Dataset2, "Length":int(Dataset2.__len__())})
print("Dataset 2 created.")


BatchSize = 3

SuperDataset = ImageDataGenerator.PyTorchMultipleDatasets(Datasets, BatchSize)
print("Super Dataset created!!!")


#print("SuperDataset Length = " + str(SuperDataset.__len__()))
#print("SuperDataset Item = " + str(SuperDataset.__getitem__(0)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warning = "" if str(device) == "cuda" else "WARNING - "
print(warning + "Device = " + str(device))



#gen = Generator.PyTorch(BatchSize, InputsNormalizer=InputsNormalizer, OutputsNormalizers=OutputsNormalizers, Workers=5)
gen = DataLoader(SuperDataset, batch_size=BatchSize, shuffle=True, num_workers=1)
print("Data loader created.")

#print("Length = " + str(gen.dataset.__len__()))

nb = 0
nbEpochs = 1
for epoch in range(nbEpochs):
	for b, batch in enumerate(gen):
		#X, Y = batch["input"].to(device, dtype=torch.float), batch["output"].to(device, dtype=torch.float)
		X, Y = batch["input"], batch["output"]
		#print(len(X))
		#print(X.shape)
		
		for i in range(X.shape[0]):
			prefix = str(epoch) + " - " + str(nb)
			ImagesIO.Write(X[i].detach().cpu().squeeze().numpy(), True, prefix + ".png")
			#ImagesIO.Write(X[i], True, prefix + ".png")
			nb += 1
		
"""























Path = "/Users/firetiti/NN/DataSets/Test CycIF 01/"
#Path = "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/Test 01/"
inputs  = Path + "/Originals x2/"
outputs = Path + "/Labels/"



CropPerImage = 1
Dimensions = 256

Generator = ImageDataGenerator.Generator(ChannelFirst=True)
Generator.setShuffle(False)
Generator.setCropPerImage(CropPerImage)
#Generator.setFlip(True)
#Generator.setRotate90x(True)
#Generator.setMaxShiftRange(10000)
Generator.setKeepEmptyOutputProbability(1.0)
Generator.LoadInputs(inputs, OnTheFly=False, Classification=False)
Generator.LoadOutputs(outputs)
Generator.setInputsDimensions(Dimensions, Dimensions)
Generator.setOutputsDimensions(Dimensions, Dimensions)
print("Generator created.")





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warning = "" if str(device) == "cuda" else "WARNING - "
print(warning + "Device = " + str(device))


"""
BatchSize = 2

gen = Generator.PyTorch(BatchSize, InputsNormalizer=InputsNormalizer, OutputsNormalizers=OutputsNormalizers, Workers=1)
print("Data loader created.")

item = gen.dataset.__getitem__(0)
X, Y = item["input"], ["output"]
print(len(X))
print(X[0].shape)
print(X.shape)
#print(Y.shape)

#sys.exit(0)

nb = 0
nbEpochs = 1
for epoch in range(nbEpochs):
	for b, batch in enumerate(gen):
		X, Y = batch["input"].to(device, dtype=torch.float), batch["output"].to(device, dtype=torch.float)
		#X, Y = batch["input"], batch["output"]
		#print(len(X))
		#print(X.shape)
		
		for i in range(X.shape[0]):
			prefix = str(epoch) + " - " + str(nb)
			ImagesIO.Write(X[i].detach().cpu().squeeze().numpy(), True, prefix + ".png")
			#ImagesIO.Write(X[i], True, prefix + ".png")
			nb += 1
"""


BatchSize = 3

Dataset = FiReTiTiPyTorchLib_Datasets.MaskRCNN(Generator, BatchSize, InputsNormalizer=InputsNormalizer, EmptyOutput=True)
Item = Dataset.__getitem__
print("Dataset created.")

DataLoader = torch.utils.data.DataLoader(Dataset, batch_size=BatchSize, shuffle=True, num_workers=5, collate_fn=utils.collate_fn)
print("Data loader created.")


nbEpochs = 1
for epoch in range(nbEpochs):
	for b, batch in enumerate(DataLoader):
		X, Y = batch["input"].to(device, dtype=torch.float), batch["output"].to(device, dtype=torch.float)



print("All done.")
