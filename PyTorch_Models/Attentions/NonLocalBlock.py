import torch

from torch import nn
from torch.nn import functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax







# ------------------------------------------------------ Non Local Block Concatenation ------------------------------------------------------
class Concatenation(nn.Module):
	
	def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
		super(Concatenation, self).__init__()

		assert dimension in [1, 2, 3]

		self.dimension = dimension
		self.sub_sample = sub_sample

		self.in_channels = in_channels
		self.inter_channels = inter_channels

		if self.inter_channels is None:
			self.inter_channels = in_channels // 2
			if self.inter_channels == 0:
				self.inter_channels = 1

		if dimension == 3:
			conv_nd = nn.Conv3d
			max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
			bn = nn.BatchNorm3d
		elif dimension == 2:
			conv_nd = nn.Conv2d
			max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
			bn = nn.BatchNorm2d
		else:
			conv_nd = nn.Conv1d
			max_pool_layer = nn.MaxPool1d(kernel_size=(2))
			bn = nn.BatchNorm1d

		self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

		if bn_layer:
			self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
									bn(self.in_channels))
			nn.init.constant_(self.W[1].weight, 0)
			nn.init.constant_(self.W[1].bias, 0)
		else:
			self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
							 kernel_size=1, stride=1, padding=0)
			nn.init.constant_(self.W.weight, 0)
			nn.init.constant_(self.W.bias, 0)

		self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

		self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0)

		self.concat_project = nn.Sequential(nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False), nn.ReLU())

		if sub_sample:
			self.g = nn.Sequential(self.g, max_pool_layer)
			self.phi = nn.Sequential(self.phi, max_pool_layer)



	def forward(self, x):
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''

		batch_size = x.size(0)

		g_x = self.g(x).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)

		# (b, c, N, 1)
		theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
		# (b, c, 1, N)
		phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

		h = theta_x.size(2)
		w = phi_x.size(3)
		theta_x = theta_x.repeat(1, 1, 1, w)
		phi_x = phi_x.repeat(1, 1, h, 1)

		concat_feature = torch.cat([theta_x, phi_x], dim=1)
		f = self.concat_project(concat_feature)
		b, _, h, w = f.size()
		f = f.view(b, h, w)

		N = f.size(-1)
		f_div_C = f / N

		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *x.size()[2:])
		W_y = self.W(y)
		z = W_y + x

		return z








# ------------------------------------------------------ Non Local Block Product ------------------------------------------------------
class Product(nn.Module):
	
	def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
		super(Product, self).__init__()

		assert dimension in [1, 2, 3]

		self.dimension = dimension
		self.sub_sample = sub_sample

		self.in_channels = in_channels
		self.inter_channels = inter_channels

		if self.inter_channels is None:
			self.inter_channels = in_channels // 2
			if self.inter_channels == 0:
				self.inter_channels = 1

		if dimension == 3:
			conv_nd = nn.Conv3d
			max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
			bn = nn.BatchNorm3d
		elif dimension == 2:
			conv_nd = nn.Conv2d
			max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
			bn = nn.BatchNorm2d
		else:
			conv_nd = nn.Conv1d
			max_pool_layer = nn.MaxPool1d(kernel_size=(2))
			bn = nn.BatchNorm1d

		self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

		if bn_layer:
			self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
									bn(self.in_channels))
			nn.init.constant_(self.W[1].weight, 0)
			nn.init.constant_(self.W[1].bias, 0)
		else:
			self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
			nn.init.constant_(self.W.weight, 0)
			nn.init.constant_(self.W.bias, 0)

		self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

		self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

		if sub_sample:
			self.g = nn.Sequential(self.g, max_pool_layer)
			self.phi = nn.Sequential(self.phi, max_pool_layer)



	def forward(self, x):
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''

		batch_size = x.size(0)

		g_x = self.g(x).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)

		theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)
		phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
		f = torch.matmul(theta_x, phi_x)
		N = f.size(-1)
		f_div_C = f / N

		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *x.size()[2:])
		W_y = self.W(y)
		z = W_y + x

		return z











# ------------------------------------------------------ Non Local Block Embedded Gaussian ------------------------------------------------------
class EmbeddedGaussian(nn.Module):
	
	def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
		super(EmbeddedGaussian, self).__init__()

		assert dimension in [1, 2, 3]

		self.dimension = dimension
		self.sub_sample = sub_sample

		self.in_channels = in_channels
		self.inter_channels = inter_channels

		if self.inter_channels is None:
			self.inter_channels = in_channels // 2
			if self.inter_channels == 0:
				self.inter_channels = 1

		if dimension == 3:
			conv_nd = nn.Conv3d
			max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
			bn = nn.BatchNorm3d
		elif dimension == 2:
			conv_nd = nn.Conv2d
			max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
			bn = nn.BatchNorm2d
		else:
			conv_nd = nn.Conv1d
			max_pool_layer = nn.MaxPool1d(kernel_size=(2))
			bn = nn.BatchNorm1d

		self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

		if bn_layer:
			self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
									bn(self.in_channels))
			nn.init.constant_(self.W[1].weight, 0)
			nn.init.constant_(self.W[1].bias, 0)
		else:
			self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
			nn.init.constant_(self.W.weight, 0)
			nn.init.constant_(self.W.bias, 0)

		self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
		self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

		if sub_sample:
			self.g = nn.Sequential(self.g, max_pool_layer)
			self.phi = nn.Sequential(self.phi, max_pool_layer)



	def forward(self, x):
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''

		batch_size = x.size(0)

		g_x = self.g(x).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)

		theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)
		phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
		f = torch.matmul(theta_x, phi_x)
		f_div_C = F.softmax(f, dim=-1)

		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *x.size()[2:])
		W_y = self.W(y)
		z = W_y + x

		return z












# ------------------------------------------------------ Non Local Block Gaussian ------------------------------------------------------
class Gaussian(nn.Module):

	def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
		super(Gaussian, self).__init__()

		assert dimension in [1, 2, 3]

		self.dimension = dimension
		self.sub_sample = sub_sample

		self.in_channels = in_channels
		self.inter_channels = inter_channels

		if self.inter_channels is None:
			self.inter_channels = in_channels // 2
			if self.inter_channels == 0:
				self.inter_channels = 1

		if dimension == 3:
			conv_nd = nn.Conv3d
			max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
			bn = nn.BatchNorm3d
		elif dimension == 2:
			conv_nd = nn.Conv2d
			max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
			bn = nn.BatchNorm2d
		else:
			conv_nd = nn.Conv1d
			max_pool_layer = nn.MaxPool1d(kernel_size=(2))
			bn = nn.BatchNorm1d

		self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

		if bn_layer:
			self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
									bn(self.in_channels))
			nn.init.constant_(self.W[1].weight, 0)
			nn.init.constant_(self.W[1].bias, 0)
		else:
			self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
			nn.init.constant_(self.W.weight, 0)
			nn.init.constant_(self.W.bias, 0)

		if sub_sample:
			self.g = nn.Sequential(self.g, max_pool_layer)
			self.phi = max_pool_layer



	def forward(self, x):
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''

		batch_size = x.size(0)

		g_x = self.g(x).view(batch_size, self.inter_channels, -1)

		g_x = g_x.permute(0, 2, 1)

		theta_x = x.view(batch_size, self.in_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)

		if self.sub_sample:
			phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
		else:
			phi_x = x.view(batch_size, self.in_channels, -1)

		f = torch.matmul(theta_x, phi_x)
		f_div_C = F.softmax(f, dim=-1)

		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *x.size()[2:])
		W_y = self.W(y)
		z = W_y + x

		return z

