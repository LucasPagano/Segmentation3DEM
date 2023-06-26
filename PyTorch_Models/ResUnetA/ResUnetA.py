import os
import sys

import torch
from torch import nn

import PyTorch_Models.Attentions
from PyTorch_Models.Attentions import Attentions


class ResUnetA(nn.Module):
    """
        This class defines a ResUNet-a model.

        Args:
            Inputs (int): The number of input image or hte number of colors in the input image.
            Depth (int): The model depth, so the number of layers/levels. Defaults to 5.
            FeatureMaps (int): The number of feature maps in the first layer of the network. Subsequent layers will be
                defined in reference to this variable. Defaults to 64.
            Activations (str): The type of activation layer to use after each convolution. "relu", "elu" and "celu" are supported, defaults to "relu".
            ResidualConnection (str): How to perform the residual connection? Must be among None or 'conv'. Default set to 'conv'.
            FirstLastBlock (str): Which convolution for the first and last block?
                Simple 1x1 convolution ("ConvBlock") or residual a-trous block ("ConvBlock"). Defaults to "ConvBlock".
            Dilations (tuple): A tuple that defines the sequence of dilations.
                For example (1,13,4) requests 4 dilations, from 1 to 13 by step of 4. Defaults to None.
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
                (convolution layer with stride 2) are supported, defaults to "stride_conv".
            UpSampling (str): The type of up sampling to use. "bilinear", "nearest", and "conv_transpose" are supported. Defaults to "nearest".
            DropOutPosition (str): Where are the drop-out layer positioned? At the end of each 'Branch' or the entire 'Block'.
            DropOut (list): A list of probablities (one per level). Defaults to None.
            OutputActivations (str): Which activation(s) use for each output? "None" (or "linear"), sigmoid", "tanh", "relu", "elu", or "celu".
                Defaults to ["sigmoid"]. The number of activations determines the number of outputs.
            OutputActivations (bool): In case of multiple outputs, should they be concatenate to make a tensor instead of a list?
    """

    def __init__(self, Inputs: int = 1, Depth: int = 6, FeatureMaps: int = 32, Activations: str = "relu",
                 ResidualConnection: str = "conv", FirstLastBlock: str = "ConvBlock", Dilations=None,
                 PSPpooling: list = None, BatchNormEncoder: list = None, BatchNormDecoder: list = None,
                 InstNormEncoder: list = None,
                 InstNormDecoder: list = None, Attention: list = None,
                 DownSampling: str = "stride_conv", UpSampling: str = "nearest", DropOutPosition: str = "Block",
                 DropOut: list = None,
                 OutputActivations: list = ["sigmoid"], ConcatenateOutputs: bool = True):
        super(ResUnetA, self).__init__()

        if DropOut is None:
            DropOut = [0.0 for i in range(Depth)]
        elif len(DropOut) != Depth:
            raise Exception("The DropOut list must have the same number of elements than the Depth value.")

        if Dilations is None:
            Dilations = [1]
        elif type(Dilations) is tuple and len(Dilations) == 3:
            Dilations = [d for d in range(Dilations[0], Dilations[1] + 1, Dilations[2])]
        else:
            raise Exception(
                "The Dilations must be a tuple of three elements: (From, To, Step). Note that To is included.")

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
        self.convright = []  # Will be converted later.
        self.down = nn.ModuleList()
        self.up = []

        if FirstLastBlock == "ConvBlock":
            self.convleft.append(ConvBlock(Inputs, FeatureMaps, activation=Activations,
                                           BatchNormalization=BatchNormEncoder[0],
                                           InstanceNormalization=InstNormEncoder[0], DropOut=DropOut[0]))
        elif FirstLastBlock == "ResBlock":
            self.convleft.append(ResUNetA_Block(Inputs, FeatureMaps,
                                                activation=Activations, dilations=Dilations,
                                                ResidualConnection=ResidualConnection,
                                                BatchNormalization=BatchNormEncoder[0],
                                                InstanceNormalization=InstNormEncoder[0],
                                                DropOutPosition=DropOutPosition, DropOut=DropOut[0]))
        else:
            raise Exception("No last block defined.")  # Cannot happen.

        self.down.append(
            DownSampleBlock(mode=DownSampling, in_fmaps=FeatureMaps, out_fmaps=FeatureMaps, activation=Activations))

        coef = 1
        for d in range(1, Depth - 1):
            self.convleft.append(ResUNetA_Block(FeatureMaps * coef, FeatureMaps * coef * 2,
                                                activation=Activations, dilations=Dilations,
                                                ResidualConnection=ResidualConnection,
                                                BatchNormalization=BatchNormEncoder[d],
                                                InstanceNormalization=InstNormEncoder[d],
                                                AttentionMecanism=Attention[d], DropOutPosition=DropOutPosition,
                                                DropOut=DropOut[d]))
            self.down.append(
                DownSampleBlock(mode=DownSampling, in_fmaps=FeatureMaps * coef * 2, out_fmaps=FeatureMaps * coef * 2,
                                activation=Activations))

            self.up.append(
                UpSampleBlock(FeatureMaps * coef * 2, FeatureMaps * coef, mode=UpSampling, activation=Activations))
            self.convright.append(ResUNetA_Block(FeatureMaps * coef * 4, FeatureMaps * coef * 2,
                                                 activation=Activations, dilations=Dilations,
                                                 ResidualConnection=ResidualConnection,
                                                 BatchNormalization=BatchNormDecoder[d],
                                                 InstanceNormalization=InstNormDecoder[d],
                                                 AttentionMecanism=Attention[d], DropOutPosition=DropOutPosition,
                                                 DropOut=DropOut[d]))
            coef *= 2

        if PSPpooling is not None:
            self.convright.insert(0, PSP(FeatureMaps * 2, FeatureMaps, sizes=PSPpooling, activation=Activations))
        elif FirstLastBlock == "ConvBlock":
            self.convright.insert(0, ConvBlock(FeatureMaps * 2, FeatureMaps, activation=Activations,
                                               BatchNormalization=BatchNormEncoder[0],
                                               InstanceNormalization=InstNormEncoder[0], DropOut=DropOut[0]))
        elif FirstLastBlock == "ResBlock":
            self.convright.insert(0, ResUNetA_Block(FeatureMaps * 2, FeatureMaps,
                                                    activation=Activations, dilations=Dilations,
                                                    ResidualConnection=ResidualConnection,
                                                    BatchNormalization=BatchNormDecoder[0],
                                                    InstanceNormalization=InstNormDecoder[0],
                                                    AttentionMecanism=Attention[0], DropOutPosition=DropOutPosition,
                                                    DropOut=DropOut[0]))
        else:
            raise Exception("No last block defined.")  # Cannot happen.

        self.up.append(
            UpSampleBlock(FeatureMaps * coef * 2, FeatureMaps * coef, mode=UpSampling, activation=Activations))

        self.convright.reverse()
        self.convright = nn.ModuleList(self.convright)
        self.up.reverse()
        self.up = nn.ModuleList(self.up)

        self.lowest = ResUNetA_Block(FeatureMaps * coef, FeatureMaps * coef * 2,
                                     activation=Activations, dilations=Dilations, ResidualConnection=ResidualConnection,
                                     BatchNormalization=BatchNormDecoder[Depth - 1],
                                     InstanceNormalization=InstNormDecoder[Depth - 1],
                                     DropOutPosition=DropOutPosition, DropOut=DropOut[Depth - 1])

        if PSPpooling is not None:
            self.PSP = PSP(FeatureMaps * coef * 2, FeatureMaps * coef * 2, sizes=PSPpooling, activation=Activations)
        else:
            self.PSP = None

        if Attention[0] == None:
            self.finalAttention = None
        elif Attention[0] == 'LAM':
            self.finalAttention = Attentions.LAMblock(FeatureMaps)
        elif Attention[0].startswith("EA_"):
            self.finalAttention = Attentions.EAblock(FeatureMaps, Attention[0])
        elif Attention[0].startswith("NL"):
            self.finalAttention = Attentions.NLblock(FeatureMaps, Attention[0])
        else:
            raise Exception("Unknown attention '" + Attention[0] + "', must be among {EA_XXX, LAM}.")

        self.final = OutBlock(FeatureMaps, OutputActivations, ConcatenateOutputs)

    def forward(self, x):
        out = []
        for d in range(len(self.down)):
            x = self.convleft[d](x)
            out.append(x)
            x = self.down[d](x)
        out.reverse()

        x = self.lowest(x)

        if self.PSP is not None:
            x = self.PSP(x)

        for d in range(len(self.up)):
            x = self.up[d](x, out[d])
            x = self.convright[d](x)

        if self.finalAttention is not None:
            x = self.finalAttention(x)

        return self.final(x)


# ------------------------------------------------------------------ ResUNetA Parts ------------------------------------------------------------------
class ResUNetA_Block(nn.Module):
    """
        This class defines the convolution block of a the ResUNet-a model.

        Args:
            in_fmaps(int) = The number of channels/feature maps to be convolved.
            out_fmaps(int) = The number of feature maps output by the convolution operation.
            activation (str): The activation function.
            dilations (list): The list of dilations (as many branches as dilations) to use in each block. Defaults to [1].
            ResidualConnection (str): How to perform the residual connection? Must be among None or 'conv'.
            BatchNormalization(bool) = Toggle batch normalization on / off. Defaults to None.
            InstanceNormalization(bool) = Toggle instance normalization on / off. Defaults to None.
            DropOutPosition (str): Where are the drop-out layer positioned? At the end of each 'Branch' or the entire 'Block'.
            DropOut (float): Use drop out? Yes if probability higher than 0.0.
            AttentionMecanism (str): Toggle attention mecanism.
    """

    def __init__(self, in_fmaps, out_fmaps, activation: str = "relu", dilations: list = [1],
                 ResidualConnection: str = "conv",
                 BatchNormalization: bool = None, InstanceNormalization: bool = None, AttentionMecanism=None,
                 DropOutPosition: str = "Block", DropOut: float = 0.0):
        super().__init__()

        self.in_fmaps = in_fmaps
        self.out_fmaps = out_fmaps

        if DropOutPosition == "Branch":
            dp = DropOut
            self.DropOut = 0.0
        elif DropOutPosition == "Block":
            dp = 0.0
            self.DropOut = DropOut
        else:
            raise Exception("DropOutPosition not supported: " + str(DropOutPosition))

        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(ResUnetA_Branch(in_fmaps, out_fmaps, dilation=d, activation=activation,
                                                 BatchNormalization=BatchNormalization,
                                                 InstanceNormalization=InstanceNormalization, DropOut=dp))

        self.ResidualConnection = ResidualConnection
        if self.ResidualConnection == "conv":
            self.resconv = nn.Conv2d(self.in_fmaps, self.out_fmaps, kernel_size=1, padding=0)
            if activation == "relu":
                self.activation = nn.ReLU(inplace=True)
            elif activation == "elu":
                self.activation = nn.ELU()
            elif activation == "celu":
                self.activation = nn.CELU()
            else:
                raise Exception("Unknown activation '" + activation + ", must be among {relu, elu, celu}.")

        if AttentionMecanism == None:
            self.Attention = None
        elif AttentionMecanism == 'LAM':
            self.Attention = Attentions.LAMblock(out_fmaps)
        elif AttentionMecanism.startswith("EA_"):
            self.Attention = Attentions.EAblock(out_fmaps, AttentionMecanism)
        elif AttentionMecanism.startswith("NL"):
            self.Attention = Attentions.NLblock(out_fmaps, AttentionMecanism)
        else:
            raise Exception("Unknown attention '" + AttentionMecanism + "', must be among {EA_*, NL*_*, LAM}.")

        if DropOut < 0.0 or 1.0 < DropOut:
            raise Exception("DropOut value is not a probability.")

    def forward(self, x):
        results = []

        if self.ResidualConnection is None:
            pass
        elif self.ResidualConnection == "conv":
            results.append(self.activation(self.resconv(x)))
        # elif self.ResidualConnection == "x2":
        #	results.append(torch.cat((x,x)))
        else:
            raise Exception(
                "Unknown residual connection '" + self.ResidualConnection + "', must be among {None, conv}.")

        for branch in self.branches:
            results.append(branch(x))

        if 1 < len(self.branches):
            stack = torch.stack(results, dim=0)
            x = stack.mean(dim=0)
        else:
            x = results[0]

        if 0.0 < self.DropOut:
            x = nn.Dropout2d(p=self.DropOut)(x)

        if self.Attention is not None:
            x = self.Attention(x)

        return x


class ResUnetA_Branch(nn.Module):
    """
        This class defines a branch (resnet block) of a the ResUNet-a block.

        Args:
            in_fmaps(int): The number of channels / feature maps to be convolved
            out_fmaps(int): The number of feature maps output by the convolution operation
            dilation (int): The convolution dilation coefficient. Defaults to 1.
            activation (str): The activation function.
            BatchNormalization(bool): Toggle batch normalization on / off. Defaults to None.
            InstanceNormalization(bool): Toggle instance normalization on / off. Defaults to None.
            DropOut (float): Use drop out? Yes if the probability is higher than 0.0
    """

    def __init__(self, in_fmaps, out_fmaps, dilation: int = 1, activation: str = "relu",
                 BatchNormalization: bool = None, InstanceNormalization: bool = None,
                 DropOut: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(out_fmaps, out_fmaps, kernel_size=3, padding=dilation, dilation=dilation)

        if activation == "relu":
            self.activation1 = nn.ReLU(inplace=True)
            self.activation2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation1 = nn.ELU()
            self.activation2 = nn.ELU()
        elif activation == "celu":
            self.activation1 = nn.CELU()
            self.activation2 = nn.CELU()
        else:
            raise Exception("Unknown activation")

        if BatchNormalization is not None:
            self.BatchNorm1 = nn.BatchNorm2d(out_fmaps, affine=BatchNormalization)
            self.BatchNorm2 = nn.BatchNorm2d(out_fmaps, affine=BatchNormalization)
        else:
            self.BatchNorm1 = self.BatchNorm2 = None

        if InstanceNormalization is not None:
            self.InstNorm1 = nn.InstanceNorm2d(out_fmaps, affine=InstanceNormalization)
            self.InstNorm2 = nn.InstanceNorm2d(out_fmaps, affine=InstanceNormalization)
        else:
            self.InstNorm1 = self.InstNorm2 = None

        if DropOut < 0.0 or 1.0 < DropOut:
            raise Exception("DropOut value is not a probability.")
        self.DropOut = DropOut

    def forward(self, x):
        x = self.conv1(x)
        if self.BatchNorm1 is not None:
            x = self.BatchNorm1(x)
        if self.InstNorm1 is not None:
            x = self.InstNorm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        if self.BatchNorm2 is not None:
            x = self.BatchNorm2(x)
        if self.InstNorm2 is not None:
            x = self.InstNorm2(x)
        x = self.activation2(x)

        if 0.0 < self.DropOut:
            x = nn.Dropout2d(p=self.DropOut)(x)

        return x


class ConvBlock(nn.Module):
    """
        This class defines a  simple convoluntional block.

        Args:
            in_fmaps(int) = The number of channels/feature maps to be convolved.
            out_fmaps(int) = The number of feature maps output by the convolution operation.
            activation (str): The activation function.
            BatchNormalization(bool): Toggle batch normalization on / off. Defaults to None.
            InstanceNormalization(bool): Toggle instance normalization on / off. Defaults to None.
            DropOut (list): Use drop out? Yes if probability higher than 0.0.
    """

    def __init__(self, in_fmaps, out_fmaps, activation: str = None, BatchNormalization: bool = None,
                 InstanceNormalization: bool = None, DropOut: float = 0.0):
        super().__init__()

        self.conv = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=1, padding=0)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "celu":
            self.activation = nn.CELU()
        else:
            raise Exception("Unknown activation: " + str(activation))

        if BatchNormalization is not None:
            self.BatchNorm = nn.BatchNorm2d(out_fmaps, affine=BatchNormalization)
        else:
            self.BatchNorm = None

        if InstanceNormalization is not None:
            self.InstNorm = nn.InstanceNorm2d(out_fmaps, affine=InstanceNormalization)
        else:
            self.InstNorm = None

        self.DropOut = DropOut

    def forward(self, x):

        x = self.conv(x)

        if self.BatchNorm is not None:
            x = self.BatchNorm(x)

        if self.InstNorm is not None:
            x = self.InstNorm(x)

        if self.activation is not None:
            x = self.activation(x)

        if 0.0 < self.DropOut:
            x = nn.Dropout2d(p=self.DropOut)(x)

        return x


class DownSampleBlock(nn.Module):
    """
        This class defines the downsampling operations.

        Args:
            in_fmaps(int): The number of channels/feature maps to be convolved.
            out_fmaps(int): The number of feature maps output by the convolution operation.
            mode(str; "max_pool", "ave_pool", or "stride_conv"): Defines which of the possible downsampling operations to use.
            activation (str): the activation function to use if the downsampling is performed using convolution.
    """

    def __init__(self, in_fmaps=None, out_fmaps=None, mode: str = "max_pool", activation: str = None):
        super().__init__()

        self.mode = mode
        if self.mode == "max_pool":
            self.down = nn.MaxPool2d(2, stride=2)
            self.activation = None
        elif self.mode == "ave_pool":
            self.down = nn.AvgPool2d(2, stride=2)
            self.activation = None
        elif self.mode == "stride_conv":
            self.down = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=1, padding=0, stride=2)
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
        This class defines the upsampling operations.

        Args:
            in_fmaps(int) = The number of channels/feature maps to be convolved.
            out_fmaps(int) = The number of feature maps output by the convolution operation.
            mode(str): Defines which upsampling operations to use among "bilinear", "nearest", or "conv_transpose". Defaults to "nearest".
    """

    def __init__(self, in_fmaps, out_fmaps, mode="nearest", activation: str = None, BatchNormalization: bool = None,
                 InstanceNormalization: bool = None):
        super().__init__()

        self.mode = mode
        if mode == "bilinear":
            self.up_sample = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            self.conv = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=1, padding=0)
        elif mode == "nearest":
            self.up_sample = nn.Upsample(scale_factor=2, mode=mode)
            self.conv = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=1, padding=0)
        elif mode == "conv_transpose":
            self.up_sample = None
            self.conv = nn.ConvTranspose2d(in_fmaps, out_fmaps, kernel_size=1, padding=0, output_padding=1, stride=2)
        else:
            raise Exception("Unknown down sampling mode.")

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "celu":
            self.activation = nn.CELU()
        else:
            raise Exception("Unknown activation: " + str(activation))

        if BatchNormalization is not None:
            self.BatchNorm = nn.BatchNorm2d(out_fmaps, affine=BatchNormalization)
        else:
            self.BatchNorm = None

        if InstanceNormalization is not None:
            self.InstNorm = nn.InstanceNorm2d(out_fmaps, affine=InstanceNormalization)
        else:
            self.InstNorm = None

    def forward(self, x, skip):
        if self.up_sample is not None:
            x = self.up_sample(x)

        x = self.conv(x)

        if self.BatchNorm is not None:
            x = self.BatchNorm(x)

        if self.InstNorm is not None:
            x = self.InstNorm(x)

        if self.activation is not None:
            x = self.activation(x)

        x = torch.cat([x, skip], dim=1)

        return x


class PSP(nn.Module):
    """
        This class defines a pyramid scene parsing pooling module.

        Args:
            in_fmaps (int): The number of input channels/feature maps to be convolved.
            out_fmaps (int): The number of feature maps / output by the convolution operation.
            sizes (list): The list of dimensions for the different AdaptiveAvgPool2d operation (one per element in the list).
            activation (str): The activation function to use.
    """

    def __init__(self, in_fmaps, out_fmaps, sizes: list = [1, 2, 3, 6], activation: str = None):
        super().__init__()

        self.stages = nn.ModuleList([self._make_stage(in_fmaps, size) for size in sizes])

        self.bottleneck = nn.Conv2d(in_fmaps * (len(sizes) + 1), out_fmaps, kernel_size=1)

        if activation == None:
            self.activation = None
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "celu":
            self.activation = nn.CELU()
        else:
            raise Exception("Output activation not supported: " + str(activation))

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [torch.nn.functional.upsample(stage(x), size=(h, w), mode="bilinear") for stage in self.stages] + [x]
        bottle = self.bottleneck(torch.cat(priors, 1))
        if self.activation == None:
            return bottle
        else:
            return self.activation(bottle)


class OutBlock(nn.Module):
    """
        This class defines the prediction procedure.

        Args:
            in_fmaps(int): The number of input channels/feature maps to be convolved.
            activations(list): The activation function(s) to use. Must match 'out_fmaps'.
            concatenate (bool): In case of multiple outputs, should they be concatenate to make a tensor instead of a list?
    """

    def __init__(self, in_fmaps, activations: list, concatenate: bool):
        super().__init__()

        self.concatenate = concatenate

        self.Convolutions = nn.ModuleList()
        self.Activations = nn.ModuleList()
        for act in activations:

            self.Convolutions.append(nn.Conv2d(in_fmaps, 1, kernel_size=1, padding=0))

            if act == None or act == "linear":
                self.Activations.append(None)
            elif act == "relu":
                self.Activations.append(nn.ReLU(inplace=True))
            elif act == "elu":
                self.Activations.append(nn.ELU())
            elif act == "celu":
                self.Activations.append(nn.CELU())
            elif act == "sigmoid":
                self.Activations.append(nn.Sigmoid())
            elif act == "tanh":
                self.Activations.append(nn.Tanh())
            else:
                raise Exception("Output activation not supported: " + str(act))

    def forward(self, x):
        outputs = []
        for conv, act in zip(self.Convolutions, self.Activations):
            y = conv(x)
            if act is not None:
                outputs.append(act(y))
            else:
                outputs.append(y)

        if len(outputs) == 1:
            return outputs[0]
        else:
            if self.concatenate == True:
                newoutputs = []
                for out in outputs:
                    newoutputs.append(out.unsqueeze(1))
                return torch.cat(newoutputs, dim=1)
            else:
                return outputs
