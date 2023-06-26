import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable





class FocalLoss(nn.Module):
	def __init__(self, gamma=0, alpha=None, size_average=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha,(float,int,long)):
			self.alpha = torch.Tensor([alpha,1-alpha])
		if isinstance(alpha,list):
			self.alpha = torch.Tensor(alpha)
		self.size_average = size_average
		
	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
			input = input.transpose(1,2) # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C
		target = target.view(-1,1)

		logpt = F.log_softmax(input)
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type()!=input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0,target.data.view(-1))
			logpt = logpt * Variable(at)
										
		loss = -1 * (1-pt)**self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()












# ------------------------------------------------------------ D1 / Dice ------------------------------------------------------------
class Dice_D1(nn.Module):
	""" Computes the classic Dice coefficient.
	"""
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Axis = Axis
		self.Smooth = Smooth

	
	def forward(self, Input, Target):
		Intersection = (Input * Target).sum(dim=self.Axis)
		cIn  = Input.sum(dim=self.Axis)
		cTar = Target.sum(dim=self.Axis)
		return (Intersection + self.Smooth) / (cIn + cTar + self.Smooth)



class Dice_D1_Dual(nn.Module):
	""" Computes the Dice loss with complement/dual.
	"""
	
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Dice_D1(Smooth=Smooth)
	
	
	def forward(self, Input, Target):
		Foreground = self.Loss(Input, Target)
		Background = self.Loss(1.-Input, 1.-Target)
		return 0.5 * (Foreground+Background)



class DiceLoss_D1(nn.Module):
	""" Computes the D1 Dice loss.
	"""
	
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Dice_D1(Axis=(1,2,3), Smooth=1.0e-5)
	
	
	def forward(self, Input, Target):
		return 1.0 - self.Loss(Input, Target).mean()



class DiceLoss_D1_Dual(nn.Module):
	""" Computes the D1 Dice loss with complement/dual.
	"""

	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Dice_D1_Dual(Axis=(1,2,3), Smooth=1.0e-5)


	def forward(self, Input, Target):
		return 1.0 - self.Loss(Input, Target).mean()











# ------------------------------------------------------------ D2 ------------------------------------------------------------
class Dice_D2(nn.Module):
	""" Computes the D2 Dice loss (squared denominator).
	"""
	
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Axis = Axis
		self.Smooth = Smooth
	
	
	def forward(self, Input, Target):
		intersection = (Input * Target).sum(dim=self.Axis)
		cIn  = torch.sum(Input**2, dim=self.Axis)
		cTar = torch.sum(Target**2, dim=self.Axis)
		return (intersection + self.Smooth) / (cIn + cTar + self.Smooth)



class Dice_D2_Dual(nn.Module):
	""" Computes the D2 Dice loss (squared denominator) with complement/dual.
	"""
	
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Dice_D2(Axis=(1,2,3), Smooth=Smooth)
	
	
	def forward(self, Input, Target):
		Foreground = self.Loss(Input, Target)
		Background = self.Loss(1.-Input, 1.-Target)
		return 0.5 * (Foreground+Background)



class DiceLoss_D2(nn.Module):
	""" Computes the D2 Dice loss.
	"""
	
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Dice_D2(Axis=(1,2,3), Smooth=1.0e-5)
	
	
	def forward(self, Input, Target):
		return 1.0 - self.Loss(Input, Target).mean()



class DiceLoss_D2_Dual(nn.Module):
	""" Computes the D2 Dice loss with complement/dual.
	"""

	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Dice_D2_Dual(Axis=(1,2,3), Smooth=1.0e-5)


	def forward(self, Input, Target):
		return 1.0 - self.Loss(Input, Target).mean()











# ------------------------------------------------------------ D3 / Tanimoto ------------------------------------------------------------
class Tanimoto(nn.Module):
	""" Computes D3/Tanimoto: D2 dice with Tanimoto coefficient.
	"""
	
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Axis = Axis
		self.Smooth = Smooth
	
	
	def forward(self, Input, Target):
		intersection = (Input * Target).sum(dim=self.Axis)
		cIn = torch.sum(Input**2, dim=self.Axis)
		cTar = torch.sum(Target**2, dim=self.Axis)
		return (intersection + self.Smooth) / (cIn + cTar - intersection + self.Smooth)



class Tanimoto_Dual(nn.Module):
	""" Computes D3/Tanimoto with complement/dual: D2 dice with Tanimoto coefficient and complement.
	"""
	
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Tanimoto(Axis=Axis, Smooth=Smooth)
	
	
	def forward(self, Input, Target):
		Foreground = self.Loss(Input, Target)
		Background = self.Loss(1.-Input, 1.-Target)
		return 0.5 * (Foreground+Background)



class DiceLoss_Tanimoto(nn.Module):
	""" Computes the D3/Tanimoto Dice loss: D2 dice loss with Tanimoto coefficient.
	"""
	
	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Tanimoto(Axis=Axis, Smooth=Smooth)
	
	
	def forward(self, Input, Target):
		return 1.0 - self.Loss(Input, Target).mean()



class DiceLoss_Tanimoto_Dual(nn.Module):
	""" Computes the D3/Tanimoto Dice loss with complement/dual: D2 dice loss with Tanimoto coefficient.
	"""

	def __init__(self, Axis=(1,2,3), Smooth=1.0e-5):
		super().__init__()
		self.Loss = Tanimoto_Dual(Axis=Axis, Smooth=Smooth)


	def forward(self, Input, Target):
		return 1.0 - self.Loss(Input, Target).mean()











# ------------------------------------------------------------ Multiple ------------------------------------------------------------
class MultipleOutputs(nn.Module):
	""" This loss is a combination of losses for models with multiple outputs.
		
		Args:
			Losses (list): The list of loss functions to use. One per output.
		
		Returns:
			The loss value.
	"""

	def __init__(self, Losses: list):
		super().__init__()
		self.Losses = Losses


	def forward(self, Outputs, Targets):
		val = 0.0
		if isinstance(Outputs, list):
			i = 0
			for output, loss in zip(Outputs, self.Losses):
				val += loss(output, Targets[:,i,:,:,:].clone())
				i += 1
		else:
			nb = Outputs.shape[1]
			for i in range(nb):
				val += self.Losses[i](Outputs[:,i,:,:,:].clone(), Targets[:,i,:,:,:].clone())
		
		return val










