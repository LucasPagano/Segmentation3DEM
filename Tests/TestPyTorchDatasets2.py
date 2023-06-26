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




InputsNormalizer255 = InputsNormalizer65535 = None
#InputsNormalizer255 = Normalizers.Basic(MaxValue=255.0)
#InputsNormalizer65535 = Normalizers.Basic(MaxValue=65535.0)
#InputsNormalizer255 = Normalizers.CenterReduce(MaxValue=255.0)
#InputsNormalizer65535 = Normalizers.CenterReduce(MaxValue=65535.0)

OutputsNormalizers = None



Dimensions = 256
Shuffle = False
BatchSize = 4


Datasets = []

Path = "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/002 - Cropped/"
inputs  = Path + "/Originals Stretched/"
outputs = Path + "/Labels/"

Generator1 = ImageDataGenerator.Generator(ChannelFirst=True)
Generator1.setShuffle(Shuffle)
Generator1.setCropPerImage(1)
Generator1.setKeepEmptyOutputProbability(1.0)
Generator1.LoadInputs(inputs, OnTheFly=False, Classification=False)
Generator1.LoadOutputs(outputs)
Generator1.setInputsDimensions(Dimensions, Dimensions)
Generator1.setOutputsDimensions(Dimensions, Dimensions)
Dataset1 = Generator1.PyTorchDataset(BatchSize, InputsNormalizers=None, OutputsNormalizers=None)
Datasets.append({"Dataset":Dataset1, "Length":Dataset1.__len__()})
print("Main Dataset created.")




Path = "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/002 - Cropped - Errors/"
inputs  = Path + "/Originals Stretched/"
outputs = Path + "/Labels/"

Generator2 = ImageDataGenerator.Generator(ChannelFirst=True)
Generator2.setShuffle(Shuffle)
Generator2.setCropPerImage(1)
Generator2.setKeepEmptyOutputProbability(1.0)
Generator2.LoadInputs(inputs, OnTheFly=False, Classification=False)
Generator2.LoadOutputs(outputs)
Generator2.setInputsDimensions(Dimensions, Dimensions)
Generator2.setOutputsDimensions(Dimensions, Dimensions)
Dataset2 = Generator2.PyTorchDataset(BatchSize, InputsNormalizers=None, OutputsNormalizers=None)
Datasets.append({"Dataset":Dataset2, "Length":1})
print("Errors Dataset created.")




Path = "/Users/firetiti/Downloads/Datasets/Auxiliary/DSB 2018/696x520/"
inputs  = Path + "/In/"
outputs = Path + "/GT/"

Generator3 = ImageDataGenerator.Generator(ChannelFirst=True)
Generator3.setShuffle(Shuffle)
Generator3.setCropPerImage(1)
Generator3.setKeepEmptyOutputProbability(0.1)
Generator3.LoadInputs(inputs, OnTheFly=False, Classification=False)
Generator3.LoadOutputs(outputs)
Generator3.setInputsDimensions(Dimensions, Dimensions)
Generator3.setOutputsDimensions(Dimensions, Dimensions)
Dataset3 = Generator3.PyTorchDataset(BatchSize, InputsNormalizers=None, OutputsNormalizers=None)
Datasets.append({"Dataset":Dataset3, "Length":4})
print("Auxiliary DSB 2018 696x520 Dataset created.")




Path = "/Users/firetiti/Downloads/Datasets/Auxiliary/DSB 2018/1272x603/"
inputs  = Path + "/In/"
outputs = Path + "/GT/"

Generator4 = ImageDataGenerator.Generator(ChannelFirst=True)
Generator4.setShuffle(Shuffle)
Generator4.setCropPerImage(1)
Generator4.setKeepEmptyOutputProbability(0.0)
Generator4.LoadInputs(inputs, OnTheFly=False, Classification=False)
Generator4.LoadOutputs(outputs)
Generator4.setInputsDimensions(Dimensions, Dimensions)
Generator4.setOutputsDimensions(Dimensions, Dimensions)
Dataset4 = Generator4.PyTorchDataset(BatchSize, InputsNormalizers=None, OutputsNormalizers=None)
Datasets.append({"Dataset":Dataset4, "Length":1})
print("Auxiliary DSB 2018 1272×603 Dataset created.")




Path = "/Users/firetiti/Downloads/Datasets/Auxiliary/S-BSST265/"
inputs  = Path + "/In/"
outputs = Path + "/GT/"

Generator5 = ImageDataGenerator.Generator(ChannelFirst=True)
Generator5.setShuffle(Shuffle)
Generator5.setCropPerImage(1)
Generator5.setKeepEmptyOutputProbability(0.1)
Generator5.LoadInputs(inputs, OnTheFly=False, Classification=False)
Generator5.LoadOutputs(outputs)
Generator5.setInputsDimensions(Dimensions, Dimensions)
Generator5.setOutputsDimensions(Dimensions, Dimensions)
Dataset5 = Generator5.PyTorchDataset(BatchSize, InputsNormalizers=None, OutputsNormalizers=None)
Datasets.append({"Dataset":Dataset5, "Length":1})
print("Auxiliary S-BSST265 Dataset created.")




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

Path = "/Users/firetiti/NN/DataSets/Test Color/"
#Path = "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/Test 01/"
inputs  = Path + "/In 1/"
outputs = Path + "/GT 1/"



CropPerImage = 2
Dimensions = 512

Generator = ImageDataGenerator.Generator(ChannelFirst=True)
Generator.setShuffle(True)
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





BatchSize = 2

gen = Generator.PyTorch(BatchSize, InputsNormalizer=InputsNormalizer, OutputsNormalizers=OutputsNormalizers, Workers=1)
print("Data loader created.")

print("Length = " + str(gen.dataset.__len__()))

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




"""
BatchSize = 3

Dataset = FiReTiTiPyTorchLib_Datasets.MaskRCNN(Generator, BatchSize, InputsNormalizer=InputsNormalizer, EmptyOutput=True)
Item = Dataset.__getitem__
print("Dataset created.")

DataLoader = torch.utils.data.DataLoader(Dataset, batch_size=BatchSize, shuffle=True, num_workers=5, collate_fn=utils.collate_fn)
print("Data loader created.")


nbEpochs = 5
for epoch in range(nbEpochs):
	for b, batch in enumerate(DataLoader):
		X, Y = batch["input"].to(device, dtype=torch.float), batch["output"].to(device, dtype=torch.float)
"""


print("All done.")
