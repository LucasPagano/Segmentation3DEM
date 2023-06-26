import os
import sys

import torch
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Attentions'))
import Attentions


class UNet(nn.Module):
	"""
	This class defines a UNet model.

	Args:
		Inputs (int): The number of input image or hte number of colors in the input image.
		Depth (int): The model depth, so the number of layers/levels. Defaults to 5.
		FeatureMaps (int): The number of feature maps in the first layer of the network. Subsequent layers will be
			defined in reference to this variable. Defaults to 64.
		KernelSize (int): The convolution kernel size. Defaults to 3.
		Activations (str): The type of activation layer to use after each convolution. "relu", "elu" and "celu" are supported, defaults to "relu".
		IntermediateActivation (bool): Use (or not) an activation function between each block convolutions. Defaults to True.
		BatchNormEncoder (list): Use batch normalization in the encoder?
			A list of elements (one per level among None, True or False for the Affine parameter). Defaults to None.
		BatchNormDecoder (list): Use batch normalization in the decoder?
			A list of elements (one per level among None, True or False for the Affine parameter). Defaults to None.
		InstNormEncoder (list): Use instance normalization in the encoder?
			A list of elements (one per level among None, True or False for the Affine parameter). Defaults to None.
		InstNormDecoder (list): Use instance normalization in the decoder?
			A list of elements (one per level among None, True or False for the Affine parameter). Defaults to None.
		Attention (list): Use attention mecanism? Must be among {EA_XXX, LAM}. Defaults to None.
		DownSampling (str): The type of down sampling to use. "max_pool" (max pooling), "ave_pool" (average pooling) and "stride_conv"
			(convolution layer with stride 2) are supported, defaults to "max_pool".
		UpSampling (str): The type of up sampling to use. "bilinear" and "conv_transpose" are supported, defaults to "bilinear".
		DropOut (list): A list of probablities (one per level). Defaults to None.
		Outputs (int): The number of predicted maps. Defaults to 1.
		OutputActivations (str): Which activation use for the output layer? Defaults to sigmoid.
	"""
	
	def __init__(self, Inputs: int=1, Depth: int=5, FeatureMaps: int=64, KernelSize: int=3, Activations: str="relu", IntermediateActivation: bool=True,
				 BatchNormEncoder: list=None, BatchNormDecoder: list=None, InstNormEncoder: list=None, InstNormDecoder: list=None, Attention: list=None,
				 DownSampling: str="max_pool", UpSampling: str="bilinear", DropOut: list=None, Outputs: int=1, OutputActivation: str="sigmoid"):
		super(UNet, self).__init__()
		
		if DropOut is None:
			DropOut = [0.0 for i in range(Depth)]
		elif len(DropOut) != Depth:
			raise Exception("The DropOut list must have the same number of elements than the Depth value.")
		
		if BatchNormEncoder is None:
			BatchNormEncoder = [None for i in range(Depth)]
		elif len(BatchNormEncoder) != Depth:
			raise Exception("The BatchNormEncoder list must have the same number of elements than the Depth value.")
		
		if BatchNormDecoder is None:
			BatchNormDecoder = [None for i in range(Depth)]
		elif len(BatchNormDecoder) != Depth:
			raise Exception("The BatchNormDecoder list must have the same number of elements than the Depth value.")
		
		if InstNormEncoder is None:
			InstNormEncoder = [None for i in range(Depth)]
		elif len(InstNormEncoder) != Depth:
			raise Exception("The InstNormEncoder list must have the same number of elements than the Depth value.")
		
		if InstNormDecoder is None:
			InstNormDecoder = [None for i in range(Depth)]
		elif len(InstNormDecoder) != Depth:
			raise Exception("The InstNormDecoder list must have the same number of elements than the Depth value.")
		
		if Attention is None:
			Attention = [None for i in range(Depth)]
		elif len(Attention) != Depth:
			raise Exception("The Attention list must have the same number of elements than the Depth value.")
		
		self.Depth = Depth
		self.convleft = nn.ModuleList()
		self.convright = [] # Will be converted later.
		self.down = nn.ModuleList()
		self.up = []
		
		self.convleft.append(ConvBlock(Inputs, FeatureMaps, kernel_size=KernelSize, activation=Activations,
									   BatchNormalization=BatchNormEncoder[0], InstanceNormalization=InstNormEncoder[0], DropOut=DropOut[0]))
		self.down.append(DownSampleBlock(mode=DownSampling, in_fmaps=FeatureMaps, out_fmaps=FeatureMaps, kernel_size=KernelSize, activation=Activations))
		
		coef = 1
		for d in range(1, Depth-1):
			self.convleft.append(ConvBlock(FeatureMaps*coef, FeatureMaps*coef*2, kernel_size=KernelSize, activation=Activations,
										   BatchNormalization=BatchNormEncoder[d], InstanceNormalization=InstNormEncoder[d],
										   DropOut=DropOut[d]))
			self.down.append(DownSampleBlock(mode=DownSampling, in_fmaps=FeatureMaps*coef*2, out_fmaps=FeatureMaps*coef*2, kernel_size=KernelSize,
											 activation=Activations))
			self.up.append(UpSampleBlock(FeatureMaps*coef*2, FeatureMaps*coef, mode=UpSampling))
			self.convright.append(ConvBlock(FeatureMaps*coef*2, FeatureMaps*coef, kernel_size=KernelSize, activation=Activations,
											BatchNormalization=BatchNormDecoder[d], InstanceNormalization=InstNormDecoder[d],
											AttentionMecanism=Attention[d], DropOut=DropOut[d]))
			coef *= 2
		
		self.up.append(UpSampleBlock(FeatureMaps*coef*2, FeatureMaps*coef, mode=UpSampling))
		self.convright.append(ConvBlock(FeatureMaps*coef*2, FeatureMaps*coef, kernel_size=KernelSize, activation=Activations,
										BatchNormalization=BatchNormDecoder[0], InstanceNormalization=InstNormDecoder[0],
										AttentionMecanism=Attention[d], DropOut=DropOut[0]))
		
		self.convright.reverse()
		self.convright = nn.ModuleList(self.convright)
		self.up.reverse()
		self.up = nn.ModuleList(self.up)

		self.lowest = ConvBlock(FeatureMaps*coef, FeatureMaps*coef*2, kernel_size=KernelSize, activation=Activations,
								BatchNormalization=BatchNormDecoder[Depth-1], InstanceNormalization=InstNormDecoder[Depth-1],
								DropOut=DropOut[Depth-1])
		
		if Attention[0] == None:
			self.finalAttention = None
		elif Attention[0] == 'LAM':
			self.finalAttention = Attentions.LAMblock(FeatureMaps)
		elif Attention[0].startswith("EA_"):
			self.finalAttention = Attentions.EAblock(FeatureMaps, Attention[0])
		else:
			raise Exception("Unknown attention '" + Attention[0] + "', must be among {EA_XXX, LAM}.")
			
		self.final = OutBlock(FeatureMaps, Outputs, activation=OutputActivation)

	
	def forward(self, x):
		out = []
		for d in range(len(self.down)):
			x = self.convleft[d](x)
			out.append(x)
			x = self.down[d](x)
		out.reverse()
		
		x = self.lowest(x)
		
		for d in range(len(self.up)):
			x = self.up[d](x, out[d])
			x = self.convright[d](x)
		
		if self.finalAttention is not None:
			x = self.finalAttention(x)
		
		return self.final(x)





# UNet Parts ----------------------------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
	"""
	This class defines the convolution block of a UNet model, and is comprised of 2 sequential padded convolutions.

	Args:
		 in_fmaps (int) = The number of channels/feature maps to be convolved
		 out_fmaps (int) = The number of feature maps output by the convolution operation
		 kernel_size (int or tuple) = The nxn dimensions of the convolutional kernel. Defaults to 3.
		 activation (str): The activation function
		 IntermediateActivation (bool): Use (or not) an activation function between the convolutions?
		 DropOut (list): Use drop out? Yes if probability higher than 0.0
		 BatchNormalization (bool): Toggle batch normalization on or off. Defaults to False.
		 InstanceNormalization (bool): Toggle instance normalization on or off. Defaults to False.
		 dropout (float) = Toggle dropout on or off. Defaults to 0.
		 AttentionMecanism (str): Toggle attention mecanism.

		 Note that batch_norm and dropout defaults must be overridden during instantiation in the UNet class
	"""

	def __init__(self, in_fmaps, out_fmaps, kernel_size=3, activation="relu", IntermediateActivation=True,
				 BatchNormalization=None, InstanceNormalization=None, AttentionMecanism=None, DropOut=0.0):
		super().__init__()
		
		self.IntermediateActivation = IntermediateActivation
		self.conv1 = nn.Conv2d(in_fmaps, out_fmaps, kernel_size, padding=int(kernel_size/2))
		self.conv2 = nn.Conv2d(out_fmaps, out_fmaps, kernel_size, padding=int(kernel_size/2))
		
		if activation == "relu":
			self.activation = nn.ReLU(inplace=True)
		elif activation == "elu":
			self.activation = nn.ELU()
		elif activation == "celu":
			self.activation = nn.CELU()
		else:
			raise Exception("Unknown activation")
		
		self.BatchNorm1 = self.BatchNorm2 = None
		if BatchNormalization is not None:
			self.BatchNorm1 = nn.BatchNorm2d(out_fmaps, affine=BatchNormalization)
			self.BatchNorm2 = nn.BatchNorm2d(out_fmaps, affine=BatchNormalization)

		self.InstNorm1 = self.InstNorm2 = None
		if InstanceNormalization is not None:
			self.InstNorm1 = nn.InstanceNorm2d(out_fmaps, affine=InstanceNormalization)
			self.InstNorm2 = nn.InstanceNorm2d(out_fmaps, affine=InstanceNormalization)
		
		if DropOut < 0.0 or 1.0 < DropOut:
			raise Exception("DropOut value is not a probability.")
		self.DropOut = DropOut
		
		if AttentionMecanism == None:
			self.Attention = None
		elif AttentionMecanism == 'LAM':
			self.Attention = Attentions.LAMblock(out_fmaps)
		elif AttentionMecanism.startswith("EA_"):
			self.Attention = Attentions.EAblock(out_fmaps, AttentionMecanism)
		else:
			raise Exception("Unknown attention '" + AttentionMecanism + "', must be among {EA_XXX, LAM}")

	def forward(self, x):
		x = self.conv1(x)
		if self.BatchNorm1 is not None:
			x = self.BatchNorm1(x)
		if self.InstNorm1 is not None:
			x = self.InstNorm1(x)
		if self.IntermediateActivation:
			x = self.activation(x)
		
		x = self.conv2(x)
		if self.BatchNorm2 is not None:
			x = self.BatchNorm2(x)
		if self.InstNorm2 is not None:
			x = self.InstNorm2(x)
		x = self.activation(x)

		if 0.0 < self.DropOut:
			x = nn.Dropout2d(p=self.DropOut)(x)

		if self.Attention is not None:
			x = self.Attention(x)
		
		return x



class DownSampleBlock(nn.Module):
	"""
	This class defines the downsampling operations used in UNet. Users can choose from a max pooling operation or
	a strided convolution. The default setup is configured for max pooling.

	Args:
		mode(str; "max_pool", "ave_pool", or "stride_conv") = Defines which of the two possible downsampling operations to use.
		in_fmaps(int) = The number of channels/feature maps to be convolved.
		out_fmaps(int) = The number of feature maps output by the convolution operation.
		kernel_size(None, int, or tuple) = The nxn dimensions of the convolutional kernel. To be overridden only for use with
		strided convolution.
		activation (str): the activation function to use if the downsampling is performed using convolution.
	"""
	
	def __init__(self, mode="max_pool", in_fmaps=None, out_fmaps=None, kernel_size=None, activation=None):
		super().__init__()
		
		self.mode = mode
		if self.mode == "max_pool":
			self.down = nn.MaxPool2d(2, stride=2)
			self.activation = None
		elif self.mode == "ave_pool":
			self.down = nn.AvgPool2d(2, stride=2)
			self.activation = None
		elif self.mode == "stride_conv":
			self.down = nn.Conv2d(in_fmaps, out_fmaps, kernel_size, padding=int(kernel_size/2), stride=2)
			if activation == "relu":
				self.activation = nn.ReLU(inplace=True)
			elif activation == "elu":
				self.activation = nn.ELU()
			elif activation == "celu":
				self.activation = nn.CELU()
			else:
				raise Exception("Unknown activation.")
		else:
			raise Exception("Unknown down sampling mode.")


	def forward(self, x):
		x = self.down(x)
		if self.activation is not None:
			x = self.activation(x)
		return x



class UpSampleBlock(nn.Module):
	"""
	This class defines the upsampling operations used in UNet. Bilinear upsampling + convolution and transpose
	convolution are supported. The default setting is configured for bilinear upsampling + convolution

	Args:
		in_fmaps(int) = The number of channels/feature maps to be convolved.
		out_fmaps(int) = The number of feature maps output by the convolution operation.
		mode(str; "bilinear" or "conv_transpose") = Defines which of the two possible upsampling operations to use.
		kernel_size(None, int, or tuple) = The nxn dimensions of the convolutional kernel.
	"""
	
	def __init__(self, in_fmaps, out_fmaps, mode="bilinear", kernel_size=3):
		super().__init__()
		self.mode = mode
		if mode == "bilinear":
			self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
			self.conv = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=kernel_size, padding=int(kernel_size/2))
		elif mode == "conv_transpose":
			self.up_sample = nn.ConvTranspose2d(in_fmaps, out_fmaps, kernel_size=kernel_size, padding=int(kernel_size/2), output_padding=1, stride=2)
		else:
			raise Exception("Unknown down sampling mode.")

	def forward(self, x, skip):
		x = self.up_sample(x)
		if self.mode == "bilinear":
			x = self.conv(x)
		x = torch.cat([x, skip], dim=1)
		return x


class OutBlock(nn.Module):
	"""
	This class defines the UNet prediction procedure. For single object segmentation, a single feature map is
	output with or without a sigmoid activation function, depending on whether or not nn.BCEWithLogitsLoss() is used.
	Multi-object segmentation allows the output of multiple feature maps. These feature maps are to be used in
	conjunction with the nn.CrossEntropyLoss() loss function.
	The default settings are configured for single channel output with a sigmoid loss function for use with
	nn.MSELoss() loss function.

	Args:
		in_fmaps(int) = The number of channels/feature maps to be convolved.
		out_fmaps(int) = The number of feature maps output by the convolution operation.

	"""
	def __init__(self, in_fmaps, out_fmaps, activation=None):
		super().__init__()
		self.conv = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=1, padding=0)
		if activation == None:
			self.activation = None
		elif activation == "relu":
			self.activation = nn.ReLU(inplace=True)
		elif activation == "elu":
			self.activation = nn.ELU()
		elif activation == "celu":
			self.activation = nn.CELU()
		elif activation == "sigmoid":
			self.activation = nn.Sigmoid()
		elif activation == "tanh":
			self.activation = nn.Tanh()
		else:
			raise Exception("Output activation not supported.")

	def forward(self, x):
		x = self.conv(x)
		if self.activation is not None:
			x = self.activation(x)
		return x
