import numpy
import os
import sys

from PIL import Image, ImageDraw
from PIL.ImageOps import autocontrast as dynexp

import torch
import torchvision
from torchvision.transforms import functional as F

sys.path.insert(0, '/Users/firetiti/Downloads/DL/TorchVision/TorchVision/detection/')
import utils
from engine import train_one_epoch, evaluate

sys.path.insert(0, './FiReTiTiPyLib/')
import Evaluations
import ImageDataGenerator
import ImagesIO
import Normalizers

sys.path.insert(0, './FiReTiTiPyLib/FiReTiTiPyTorchLib/')
import FiReTiTiPyTorchLib_Datasets as Datasets
import FiReTiTiPyTorchLib_Evaluations as Evaluations
import FiReTiTiPyTorchLib_Segmentator as Segmentator

sys.path.insert(0, '/Users/firetiti/NN/PyTorch/Models/MaskRCNN/')
import MaskRCNN





# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
warning = "\n" if str(device) == "cuda" else "\nWARNING - "
print(warning + "Device = " + str(device) + "\n")



#model = torch.load("./PyTorch/MaskRCNN_256x256_Norm=B - Baseline.pt", map_location=torch.device('cpu'))
#model = torch.load("./PyTorch/MaskRCNN_256x256_Norm=CR - 964_812_873 - 20200619.pt", map_location=torch.device('cpu'))
model = torch.load("./PyTorch/MaskRCNN_512x512_Norm=CR - 967_821_884 - 20200624.pt", map_location=torch.device('cpu'))
#model = torch.load("./PyTorch/MaskRCNN_512x512.pt", map_location=torch.device('cpu'))

MaskRCNN.Trainable(model, "backbone", False)
MaskRCNN.Trainable(model, 'rpn', False)
MaskRCNN.Trainable(model, "roi", True)
MaskRCNN.Trainable(model, "rpn", True)
MaskRCNN.Trainable(model, "backbone", True)

sys.exit(0)





model.eval()
model = model.to(device)



#inputs = "/Users/firetiti/Downloads/CyclicIF/Originals Stretched 03/"
#inputs = "./DataSets/Cyclic IF 1/"
#inputs = "./DataSets/Cyclic IF 2/"
inputs = "./DataSets/Cyclic IF 3/"
#inputs = "./DataSets/Test CycIF 01/Originals/"
dim = 512
InputNormalizer = Normalizers.Basic(MaxValue=65535.0)
BatchSize = 3
BorderEffectSize = 73
CheckFullOverlap = 7


Threshold = 0.19
segmentator = Segmentator.Segment()
segmentator.MaskRCNN(inputs, ".png", model, device, CropSize=dim, BorderEffectSize=BorderEffectSize, InputNormalizer=InputsNormalizer,
					BatchSize=BatchSize, CheckFullOverlap=CheckFullOverlap, ResultsDirPath="./Segmentator " + str(Threshold) + "/",
					SaveIndividualObject=False, Threshold=Threshold, Margin=1)
segmentator = None
print("Segmentation Done!\n\n")
sys.exit(0)


eval = Evaluations.MaskRCNN()
eval.Directory("./Segmentator "+str(Threshold)+"/", "./DataSets/Test CycIF 01/Labels/", Threshold=128, Overlap=0.51, SaveImagesIn="./Results/", SaveEvalutationsAs="./Results/Evaluations.csv", SegmentationOnly=False)
#Acc, Sensitivity, Specificity, Precision, Dice, Acc2, Sensitivity2, Precision2 = eval.EvaluateImage(segmentation, segGT, labels, labelsGT, Threshold=128, Overlap=0.51, SaveImagesAs="./Results/Test")
sys.exit(0)


"""
Acc, Sensitivity, Specificity, Precision, Dice = eval.Segmentation(segmentation, segGT, SaveImageAs="./Results/Segmentation.png")
print("Accuracy = " + str(Acc))
print("Sensitivity = " + str(Sensitivity))
print("Specificity = " + str(Specificity))
print("Precision = " + str(Precision))
print("Dice = " + str(Dice))


Acc, Sensitivity, Precision = eval.Detection(labels, labelsGT, Overlap=0.51, SaveImageAs="./Results/Detections.png")
print("Accuracy = " + str(Acc))
print("Recall = " + str(Sensitivity))
print("Precision = " + str(Precision))
"""
sys.exit(0)

