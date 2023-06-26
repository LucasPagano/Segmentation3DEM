import numpy as np
import os
import sys
import time

from PIL import Image

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
# from torchsummary import summary

# from torchviz import make_dot

sys.path.insert(0, '../')
import Evaluations
import ImagesIO
import ImageDataGenerator
import JavaInterfacer
import Metrics
import Normalizers
import Processing

sys.path.insert(0, '../FiReTiTiPyTorchLib/')
import FiReTiTiPyTorchLib_Evaluations as Evaluations
import FiReTiTiPyTorchLib_Losses as Losses
import FiReTiTiPyTorchLib_Schedulers as Schedulers
import FiReTiTiPyTorchLib_Segmentator as Segmentator

sys.path.insert(0, '../PyTorch_Models/ResUnet-a/')
import ResUnetA

'''
sys.path.insert(0, './PyTorch/Models/U-Net/')
import UNet

sys.path.insert(0, './PyTorch/Models/Denoising/')
import Denoising

sys.path.insert(0, './PyTorch/Models/MaskRCNN/')
import MaskRCNN

import bisect
'''


def main():
	"""
	model = ResUnetA.ResUnetA(Inputs=1, Depth=6, FeatureMaps=32, Activations="relu",
							ResidualConnection="conv", FirstLastBlock="ResBlock", Dilations=(1,13,6),
							DownSampling="max_pool", UpSampling="nearest",
							#BatchNormEncoder=[True, True, True, True, True, True], BatchNormDecoder=[True, True, True, True, True, True],
							#InstNormEncoder=[True, True, True, True, True, True], InstNormDecoder=[True, True, True, True, True, True],
							BatchNormEncoder=[True, True, True, True, True, True], InstNormDecoder=[True, True, True, True, True, True],
							DropOut=None, OutputActivations=["sigmoid", "sigmoid", "sigmoid", "sigmoid"], ConcatenateOutputs=False)
	"""
	model = ResUnetA.ResUnetA(Inputs=1, Depth=3, FeatureMaps=4, Activations="relu",
					ResidualConnection="conv", FirstLastBlock="ResBlock", Dilations=(1,13,12),
					DownSampling="max_pool", UpSampling="nearest", #PSPpooling=[1, 3, 7],
					#BatchNormEncoder=[True, True, True, True, True, True], BatchNormDecoder=[True, True, True, True, True, True],
					#InstNormEncoder=[True, True, True, True, True, True], InstNormDecoder=[True, True, True, True, True, True],
					BatchNormEncoder=[True, True, True], InstNormDecoder=[True, True, True],
					#Attention=['EA_64_1_64_BN_0', 'EA_64_1_64_IN_0', 'EA_64_1_64_None_None'],
					#Attention=['NLE_None_2_True_True', None, None],
					DropOutPosition="Block", DropOut=None, OutputActivations=["sigmoid", "sigmoid", "sigmoid", "sigmoid"], ConcatenateOutputs=True)
					#DropOutPosition="Block", DropOut=None, OutputActivations=["sigmoid"], ConcatenateOutputs=False)


	"""
	model = UNet.UNet(Inputs=1, Depth=4, FeatureMaps=8, KernelSize=3, Activations="relu", IntermediateActivation=False,
					DownSampling="max_pool", UpSampling="bilinear",
					BatchNormEncoder=[True, True, True, True], BatchNormDecoder=[True, True, True, True],
					#InstNormEncoder=[True, True, True, True], InstNormDecoder=[True, True, True, True],
					DropOut=[0.13, 0.11, 0.07, 0.05], Attention=['EA_16_4_16', 'EA_16_4_16', 'EA_16_4_16', 'EA_16_4_16'],
					Outputs=4, OutputActivation="sigmoid")
	"""

	#print(model)
	#summary(model, (1, 256, 256))

	#sys.exit(0)
	"""
	model = Denoising.DenoiseNet(Depth=5, FeatureMaps=12)
	print(model)
	summary(model, (1, 256, 256))
	sys.exit(0)
	"""
	#loss_fn = torch.nn.SmoothL1Loss(size_average=False)
	#loss_fn = torch.nn.MSELoss(size_average=False)
	#loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
	#loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
	#loss_fn = torch.nn.BCELoss(size_average=True)
	#loss_fn = Losses.DiceLoss_D1_Dual()
	#loss_fn = Losses.DiceLoss_D2_Dual()
	#loss_fn = Losses.DiceLoss_Tanimoto_Dual()
	#loss_fn = Losses.MultipleOutputs([Losses.DiceLoss_D1_Dual(Smooth=1), Losses.DiceLoss_D1_Dual(Smooth=1)])
	loss_fn = Losses.MultipleOutputs([Losses.DiceLoss_Tanimoto_Dual(), Losses.DiceLoss_Tanimoto_Dual(), Losses.DiceLoss_Tanimoto_Dual(), Losses.DiceLoss_Tanimoto_Dual()])

	optimizer = torch.optim.Adam(model.parameters())
	#optimizer = torch.optim.Adadelta(model.parameters(), lr=0.9, rho=0.95, weight_decay=0.001)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	warning = "" if str(device) == "cuda" else "WARNING - "
	print(warning + "Device = " + str(device))
	#model = model.to(device)

	#sys.exit(0)

	#inputs = "./Data/Input/"
	#outputs = "./Data/ADM/"
	#inputs  = "./DataSets/Test Inputs Gray/"
	#outputs = "./DataSets/Test Outputs/"
	#outputs = "./DataSets/Test Outputs x2/"
	#inputs  = "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/Results/Originals/"
	#outputs = "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/Results/GT0/Foreground/"
	inputs  = "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/002 - Cropped - Small/Originals Stretched/"
	outputs = "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/002 - Cropped - Small/Outs 01/"


	dim = 256
	minus = 0
	BatchSize = 10
	nbCropPerImage = 2

	#InputsNormalizers  = Normalizers.Normalize(MaxValue=65535)
	#InputsNormalizers  = Normalizers.Normalize()
	InputsNormalizers  = [Normalizers.CenterReduce()]
	#InputsNormalizers  = Normalizers.CenterReduceGlobal(RemoveOutliers=0)
	OutputsNormalizers = [Normalizers.Basic(MaxValue=30.0), Normalizers.Basic(MaxValue=255.0), Normalizers.Basic(MaxValue=65535.0), Normalizers.Basic(MaxValue=255.0)]
	#OutputsNormalizers = Normalizers.Basic()

	generator = ImageDataGenerator.Generator(ChannelFirst=True)
	generator.setShuffle(True)
	generator.setCropPerImage(nbCropPerImage)
	generator.setFlip(True)
	generator.setRotate90x(True)
	#generator.setMaxShiftRange(10)
	#generator.setBrighterDarker(13, 'Uniform_PerChannel')
	#generator.setNoise(3, 'Gaussian', 17)
	generator.setKeepEmptyOutputProbability(0.23)
	generator.LoadInputs(inputs, OnTheFly=False, Classification=False)
	generator.LoadOutputs(outputs)
	generator.setInputsDimensions(dim, dim)
	generator.setOutputsDimensions(dim-minus, dim-minus)
	dl = generator.PyTorch(BatchSize, InputsNormalizers=InputsNormalizers, OutputsNormalizers=OutputsNormalizers, Workers=0)


	"""
	# An example to display the network architecture.
	batch = next(iter(dl))
	input, output = batch["input"].to(device, dtype=torch.float), batch["output"].to(device, dtype=torch.float)

	dot = make_dot(model(input), params=None)#params=dict(model.named_parameters()))
	dot.render('Model', view=True)
	sys.exit(0)
	"""

	#scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
	#milestones = {3 : 0.0005, 5 : 0.0001, 8 : 0.00005, 10 : 0.00001}
	#scheduler = Schedulers.CustomMultiStepLR(optimizer, baselr=0.001, milestones=milestones)
	#scheduler = Schedulers.MyMultiStepLR(optimizer, milestones=[3, 5, 8], gamma=0.1)
	
	losshistory = []
	nbepochs = 13
	for epoch in range(nbepochs):
		lossvalue = 0.0
		start = time.time()
		for b, batch in enumerate(dl):
			X, Y = batch["input"].to(device, dtype=torch.float), batch["output"].to(device, dtype=torch.float)
			#Y = Y.squeeze()
			#print(X.shape)
			#print(Y.shape)
			
			optimizer.zero_grad() # Zero the gradients before running the backward pass.
			
			Ypred = model(X) # Forward pass: compute predicted y
			#print(Ypred.shape)
			#Ypred = Ypred[:,:,np.newaxis,:,:]
			#print(Ypred.shape)
			loss = loss_fn(Ypred, Y) # Compute and print loss
			lossvalue += loss.item()
			loss.backward()
			
			optimizer.step()
		end = time.time()
		#print("Epoch %d - loss = %f - %f s - LR=%013.12f" % (epoch, lossvalue, (end-start), scheduler.get_lr()[0]))
		print("Epoch %d - loss = %f - %f s" % (epoch, lossvalue, (end-start)))

		"""for param_group in optimizer.param_groups:
			print("Epoch " + str(epoch) + ", LR = " + str(param_group["lr"]))
		scheduler.step()"""
		losshistory.append(lossvalue)
	print("Training Done!")

	sys.exit(0)



	torch.save(model, "Model_ADM.pt")

	with open("Loss.txt", "w") as f:
		for l in losshistory:
			f.write(str(l) +"\n")

	sys.exit(0)

	model.eval()

	"""
	generator = ImageDataGenerator.Generator(ChannelFirst=True)
	generator.LoadInputs(inputs, OnTheFly=False, Classification=False)
	generator.LoadOutputs(outputs)
	generator.setInputsDimensions(dim, dim)
	generator.setOutputsDimensions(dim, dim)
	dl = generator.PyTorch(BatchSize, InputsNormalizer=Processing.Normalize, OutputsNormalizer=Processing.NormalizeBasic, Workers=4)
	"""
	imnames = ImagesIO.FindImages(inputs)
	names = list(np.repeat(imnames, nbCropPerImage))
	imnames = names

	nbEpoch = 2
	for epoch in range(nbEpoch):
		index = 0
		for b, batch in enumerate(dl):
			X, Y = batch['input'].to(device), batch['output'].to(device)
			print("X => " + str(X.shape))
			print("Y => " + str(Y.shape))
			Z = model(X)
			print("Z => " + str(Z.shape))
			sys.exit(0)
			#print("Z => " + str(Z.shape) + " => " + str(torch.min(Z)) + " & " + str(torch.max(Z)))
			
			Processing.Denormalize16bits(X)
			Processing.DenormalizeBasic(Y)
			#Processing.DenormalizeBasic(Z)
			
			for i in range(X.shape[0]):
				#print("Index Out = %d" % index)
				prefix = str(epoch) + " - " + os.path.basename(imnames[index]) + " - " + str(index)
				#ImagesIO.Write(X[i].detach().cpu().squeeze().numpy(), False, "./Results/In/Epoch " + prefix + " - Input.png")
				#ImagesIO.Write(Y[i].detach().cpu().squeeze().numpy(), False, "./Results/GT/Epoch " + prefix + " - GroundTruth.png")
				#ImagesIO.Write(Z[i].detach().cpu().squeeze().numpy(), True, "./Results/Out/Epoch " + prefix + " - Prediction.png")
				#print(Z[i].detach().cpu().squeeze().numpy()[0:5, 0:5])
				index += 1
	print("Predictions Done!")



if __name__ == '__main__':
	print("Cocorico")
	main()



print("Done!")
sys.exit(0)








"""
ev = Evaluations.Evaluations(Threshold=128)
ev.BestSegmentation2("./Test 2 Out/", "./Test 2 GT/", SaveEvalutationsAs="./Results/Evaluations2/Evaluations.csv", SaveEvalutationImagesIn="./Results/Evaluations2/")
ev.BestSegmentation("./Test 2 Out/", "./Test 2 GT/", SaveEvalutationsAs="./Results/Evaluations/Evaluations.csv", SaveEvalutationImagesIn="./Results/Evaluations/")
print("Evaluations Done!")
print("Done\n\n")
sys.exit(0)




device = "cpu"

model = torch.load("./Model_0.700461.pt", map_location="cpu")
model.eval()
model = model.to(device)

InputsNormalizer = Normalizers.Normalize(MaxValue=65535)
OutputsNormalizer = Normalizers.Basic()

segmentator = Segmentator.Segment()
segmentator.PyTorch("/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/Test", model, device, CropSize=512, BorderEffectSize=31,
InputNormalizer=InputsNormalizer, BatchSize=5, OutputNormalizer=OutputsNormalizer,
ResultsDirPath="./Results/Segmentator/")


print("Done\n\n")
sys.exit(0)
"""
