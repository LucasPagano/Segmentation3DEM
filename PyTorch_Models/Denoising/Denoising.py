import torch

from torch import nn


class DenoiseNet(nn.Module):
	"""
	This class defines a denoising model.

	Args:
		Inputs (int): The number of input image or hte number of colors in the input image.
		Depth (int): The model depth, so the number of layers/levels. Defaults to 24.
		FeatureMaps (int): The number of feature maps per layer/level. Defaults to 64.
		KernelSize (int): The convolution kernel size. Defaults to 3.
		Activations (str): The type of activation layer to use after each convolution. "relu" and "elu" are supported, defaults to "relu".
		BatchNormalization (bool): Use batch normalization? Defaults to False.
		DropOut (float): The drop out probability. Must be in range [0,1], and defaults to 0.
		Outputs (int): The number of predicted maps. Defaults to 1.
	"""
	
	def __init__(self, Inputs: int=1, Depth: int=24, FeatureMaps: int=64, KernelSize: int=3, Activations: str="relu",
				BatchNormalization: bool=None, DropOut: float=0.0, Outputs: int=1):
		super(DenoiseNet, self).__init__()
		
		if DropOut < 0.0 or 1.0 < DropOut:
			raise Exception("DropOut value must be in range [0,1].")
		
		self.Depth = Depth
		self.DropOut = DropOut
		
		self.FirstResConv = nn.Conv2d(Inputs, Outputs, KernelSize, padding=int(KernelSize/2))
		
		self.MainConv = nn.ModuleList()
		self.MainConv.append(nn.Conv2d(Inputs, FeatureMaps-1, KernelSize, padding=int(KernelSize/2)))
		
		self.ResConv = nn.ModuleList()
		self.ResConv.append(nn.Conv2d(FeatureMaps-1, Outputs, KernelSize, padding=int(KernelSize/2)))
		
		for d in range(1, self.Depth):
			self.MainConv.append(nn.Conv2d(FeatureMaps-1, FeatureMaps-1, KernelSize, padding=int(KernelSize/2)))
			self.ResConv.append(nn.Conv2d(FeatureMaps-1, Outputs, KernelSize, padding=int(KernelSize/2)))

		if Activations == "relu":
			self.Activation = nn.ReLU(inplace=True)
		elif Activations == "elu":
			self.Activation = nn.ELU()
		else:
			raise Exception("Unknown activation")

		if BatchNormalization is not None:
			self.BatchNormalization = nn.BatchNorm2d(FeatureMaps-1, affine=BatchNormalization)
		else:
			self.BatchNormalization = None



	def forward(self, x):
		result = self.FirstResConv(x)
		
		for d in range(0, self.Depth):
			x = self.MainConv[d](x)
			if self.BatchNormalization is not None:
				x = self.BatchNormalization(x)
			if 0.0 < self.DropOut:
				x = nn.Dropout2d(p=self.DropOut)(x)
			x = self.Activation(x)
			result += self.ResConv[d](x)
		
		return result
