import numpy as np
import random
import scipy.ndimage
import sys

import ImagesIO



#FastNoise = True
GaussStdNoise = 0.5102041 # 99.7% within [–1,1]
#GaussStdNoise = 0.3333333 # 95% within [–1,1]
Poisson = 31.0 # 99.9% within [–1,1]
#Poisson = 23.0 # 99% within [–1,1]

SafetyCount = 200





class Transformations:

	def __init__(self, MaxShift: int, Flip: bool, Rotate90: bool, Rotate: bool, AngleRange: int, AngleStep: int, RotateMode: str, FillingValues: list,
				 nbPixelBrighterDarker: int, BrighterDarkerType: str, KeepEmptyOutput: float, Normalizers):
		""" This function performs the exact same transformation to the inputs and the outputs. However, the outputs intensity is not modified.
			Args:
				MaxShift (int): The maximum shift range that the crop can be cut from the image center.
				Flip (bool): Is the image need to be randomly flipped?
				Rotate90 (bool): Is the image need to be randomly rotated by a multiple of 90 degrees?
				Rotate (bool): Is the image need to be randomly rotated?
				AngleRange (int): The maximum rotation possible. The random rotation will be in [-AngleRange,AngleRange]
				AngleStep (int): The rotation increment.
				RotateMode (str): How to fill the empty pixels if the rotation is active.
				FillingValues (list): The filling values to use if Rotate is True and the RotateMode is 'constant'.
				nbPixelBrighterDarker (int):  The modification range. Each channel will be modified by a value in the range
					[-nbPixelBrighterDarker, nbPixelBrighterDarker]
				BrighterDarkerType (str): The modification mode. See dedicated set function for the options.
				KeepEmptyOutput (float): The probability to keep an empty/black output.
				Normalizers: The normalizer used on the input images.
		"""
		self.MaxShift = MaxShift
		self.Flip = Flip
		self.Rotate90 = Rotate90
		self.Rotate = Rotate
		self.AngleRange = AngleRange
		self.AngleStep = AngleStep
		self.RotateMode = RotateMode
		self.FillingValues = FillingValues
		self.nbPixelBrighterDarker = nbPixelBrighterDarker
		self.BrighterDarkerType = BrighterDarkerType
		self.KeepEmptyOutput = KeepEmptyOutput
		self.Normalizers = Normalizers










	def Crop(self, x, rand, SizeX: int, SizeY: int, CropSizeX: int, CropSizeY: int, MaxShift: int, ChannelFirst: bool):
		""" This function performs a random crop of the image.
			Args:
				x: The image to crop.
				rand: The random class to use.
				SizeX (int): The image width.
				SizeY (int): The image height.
				CropSizeX (int): The resulting image width (the crop X dimension).
				CropSizeY (int): The resulting image height (the crop Y dimension).
				MaxShift (int): The maximum range/distance that the crop can be taken from the image center.
				ChannelFirst (bool): Where are the image channels/colors located?
		"""
		startx = int((SizeX-CropSizeX) / 2)
		shiftx = min(SizeX-CropSizeX, 2*MaxShift)
		randminx = max(0, startx-shiftx)
		xshift = rand.randint(randminx, randminx+shiftx)
	
		starty = int((SizeY-CropSizeY) / 2)
		shifty = min(SizeY-CropSizeY, 2*MaxShift)
		randminy = max(0, starty-shifty)
		yshift = rand.randint(randminy, randminy+shifty)
		
		if ChannelFirst == True:
			return np.copy(x[:, yshift:yshift+CropSizeY, xshift:xshift+CropSizeX])
		else:
			return np.copy(x[yshift:yshift+CropSizeY, xshift:xshift+CropSizeX, :])



	def Rotation90x(self, x, rand, ChannelFirst: bool):
		""" This function performs a random rotation of the image by a multiple of 90 degrees.
			Args:
				x: The image to rotate.
				rand: The random class to use.
				ChannelFirst (bool): Where are the image channels/colors located?
		"""
		randomvalue = rand.randint(0,3)
		if randomvalue == 0:
			return x
	
		if ChannelFirst == True:
			nbchannel = x.shape[0]
			for i in range(nbchannel):
				x[i] = np.rot90(x[i], randomvalue)
		else:
			nbchannel = x.shape[2]
			for i in range(nbchannel):
				x[:, :, i] = np.rot90(x[:, :, i], randomvalue)
		return x



	def _transform_matrix_offset_center(self, matrix, x, y):
		o_x = float(x) / 2 + 0.5
		o_y = float(y) / 2 + 0.5
		offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
		reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
		transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
		return transform_matrix

	def Rotation(self, x, rand, ChannelFirst: bool):
		""" This function performs a random rotation of the image.
			Args:
				x: The image to rotate.
				rand: The random class to use.
				ChannelFirst (bool): Where are the image channels/colors located?
		"""
		raise Exception("Does not work, must be upgraded.")
		range = int(self.AngleRange / self.AngleStep)
		randomangle = rand.randint(-range, range) * self.AngleStep
		if randomangle == 0:
			return x
	
		if ChannelFirst == True:
			row_axis = 1
			col_axis = 2
			channel_axis = 0
		else:
			row_axis = 0
			col_axis = 1
			channel_axis=2

		theta = np.deg2rad(randomangle)
		rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
									[np.sin(theta), np.cos(theta), 0],
									[0, 0, 1]])
	
		h, w = x.shape[row_axis], x.shape[col_axis]
		transform_matrix = self._transform_matrix_offset_center(rotation_matrix, h, w)
		x = np.rollaxis(x, channel_axis, 0)
		final_affine_matrix = transform_matrix[:2, :2]
		final_offset = transform_matrix[:2, 2]
		
		channel_images = []
		c = 0
		while c < x.shape[0]:
			channel_images.append(scipy.ndimage.affine_transform(x[c], final_affine_matrix, final_offset, order=1, mode=self.RotateMode, cval=0))
			c += 1
		
		x = np.stack(channel_images, axis=0)
		x = np.rollaxis(x, 0, channel_axis + 1)
	
		return x



	def Flips(self, x, rand, axis):
		""" This function performs a random rotation of the image.
			Args:
				x: The image to flip.
				rand: The random class to use.
				axis (int): The flipping axis.
		"""
		if rand.randint(0,1) == 0:
			return x
		x = np.asarray(x).swapaxes(axis, 0)
		x = x[::-1, ...]
		x = x.swapaxes(0, axis)
		return x



	def BrighterDarker(self, x, rand, Channels: int, ChannelFirst: bool):
		""" This function modifies the overall image brightness.
			Args:
				x: The image to modify.
				rand: The random class to use.
				Channels (int): The image number of channels.
				ChannelFirst (bool): Is the color channel placed first in the images?
				nbPixelBrighterDarker (int): The modification range. Each channel will be modified by a value in the range
					[-nbPixelBrighterDarker, nbPixelBrighterDarker]
				BrighterDarkerType (str): The modification mode (see below).
				Normalizer: The normalizer used on the input images.
		
			The parameter BrighterDarkerType can be:
				- Uniform_Equal = unique random value added/subtracted to every channels.
				- Uniform_PerChannel = one random value added/subtracted per channel.
				- Gaussian_Equal = unique random value from gaussian distribution added/subtracted to every channels.
				- Gaussian_PerChannel = one random value from gaussian distribution added/subtracted per channel.
		"""
		values = []
		if self.BrighterDarkerType == "Uniform_Equal":
			v = rand.uniform(-self.nbPixelBrighterDarker, self.nbPixelBrighterDarker)
			for c in range(Channels):
				values.append(v)
		elif self.BrighterDarkerType == "Uniform_PerChannel":
			for c in range(Channels):
				values.append(rand.uniform(-self.nbPixelBrighterDarker, self.nbPixelBrighterDarker))
		elif self.BrighterDarkerType == "Gaussian_Equal":
			v = rand.gauss(0, GaussStdNoise) * self.nbPixelBrighterDarker
			for c in range(Channels):
				values.append(v)
		elif self.BrighterDarkerType == "Gaussian_PerChannel":
			for c in range(Channels):
				values.append(rand.gauss(0, GaussStdNoise) * self.nbPixelBrighterDarker)
		else:
			raise NameError("The brighter/darker type must be Uniform_Equal, Uniform_PerChannel, Gaussian_Equal, or Gaussian_PerChannel: "
							+ self.BrighterDarkerType)

		if self.Normalizers is not None:
			for c in range(Channels):
				values[c] /= self.Normalizers[c].getMaxValue() / 2.0

		if ChannelFirst == True:
			for c in range(Channels):
				x[c] += values[c]
		else:
			for c in range(Channels):
				x[:, :, c] += values[c]


	"""
	@numba.njit
	def RandomNoiseFast(x, noisetype, nbpixelnoise, frequency, maxvalue=255.0):
		x = x.ravel()
		const = nbpixelnoise / maxvalue
	
		if noisetype == 1:
			for ii in range(len(x)):
				if np.random.randint(0, 100) <= frequency:
					x[ii] += np.random.normal(0, GaussStdNoise) * const
		elif noisetype == 2:
			const *= 2.0
			for ii in range(len(x)):
				if np.random.randint(0, 100) <= frequency:
					x[ii] += (np.random.random()- 0.5) * const
		elif noisetype == 3:
			for ii in range(len(x)):
				if np.random.randint(0, 100) <= frequency:
					x[ii] += float(np.random.poisson(100)-101) / Poisson * const
		else:
			raise NameError("The noise type must be 1 (Gaussian), 2 (Uniform), or 3 (Poisson).")
	"""




	def Noise(self, x, Range: int, Type: str, Frequency: int, Normalizer):
		""" This function add noise.
			Args:
				x: The image to transform.
				Range (int): The number/range of pixels in which X% of the noise values will be.
				Type (str): The nois type. Must be in Gaussian, Uniform, or Poisson.
				Frequency (int): The noise frequency is the percentage of pixels to modify with noise.
				Normalizer: How to normalize the values.
		"""
		raise Exception("Must be updated to supoprt Multiple input normalizers.")
		randomvalues = np.random.randint(0, 100, size=x.shape)
		randomvalues[randomvalues <= Frequency] = 1
		randomvalues[Frequency < randomvalues] = 0
		
		if Type == "Gaussian":
			noise = np.random.normal(0, GaussStdNoise, x.shape) * randomvalues * Range
		elif Type == "Uniform":
			noise = (np.random.random(x.shape) - 0.5) * randomvalues * 2.0 * Range
		elif Type == "Poisson":
			noise = np.random.poisson(100, x.shape) - 101
			noise = noise.astype(float) / Poisson
			noise = noise * randomvalues * Range
		else:
			raise NameError("The noise type must be Gaussian, Uniform, or Poisson:" + Type)
		
		if self.Normalizer is not None:
			noise /= Normalizer.getMaxValue() / 2.0
		
		x += noise




	def TransformSingle(self, x, Channels: int, ChannelFirst: bool, CropSizeX: int, CropSizeY: int, isInput: bool):
		""" This function manages the various random transformations.
			Args:
				x: The image to transform.
				Channels (int): The image number of channels.
				ChannelFirst (bool): Is the color channel placed first in the images?
				CropSizeX (int): The crop width.
				CropSizeY (int): The crop height.
				isInput (bool): Is the data to process an input? If True, then the brighter/darker transformation will be called.
		"""
		SizeX, SizeY, channels = self._FindDimensions(x, ChannelFirst)

		return self.Transform(x, random.Random(), SizeX, SizeY, Channels, ChannelFirst, CropSizeX, CropSizeY, isInput)
	
	
	def Transform(self, x, rand, SizeX: int, SizeY: int, Channels: int, ChannelFirst: bool, CropSizeX: int, CropSizeY: int, isInput: bool):
		""" This function manages the various random transformations.
			Args:
				x: The image to transform.
				rand: The random class to use.
				SizeX (int): The image width.
				SizeY (int): The image height.
				Channels (int): The image number of channels.
				ChannelFirst (bool): Is the color channel placed first in the images?
				CropSizeX (int): The crop width.
				CropSizeY (int): The crop height.
				isInput (bool): Is the data to process an input? If True, then the brighter/darker transformation will be called.
		"""
		x = self.Crop(x, rand, SizeX, SizeY, CropSizeX, CropSizeY, self.MaxShift, ChannelFirst)
		
		if self.Flip == True:
			x = self.Flips(x, rand, 1)
			if ChannelFirst == True:
				x = self.Flips(x, rand, 2)
			else:
				x = self.Flips(x, rand, 0)
	
		if self.Rotate90 == True:
			x = self.Rotation90x(x, rand, ChannelFirst)
		elif self.Rotate == True:
			x = self.Rotation(x, rand, ChannelFirst)
		
		if isInput and 0 < self.nbPixelBrighterDarker:
			self.BrighterDarker(x, rand, Channels, ChannelFirst)

		return x


	def TransformUnison(self, Inputs: list, Outputs: list, Channels: int, ChannelFirst: bool,
							  InCropSizeX: int, InCropSizeY: int, OutCropSizeX: int, OutCropSizeY: int):
		""" This function performs the exact same transformation to the inputs and the outputs. However, the outputs intensity is not modified.
			Args:
				Inputs (list): The input images to transform.
				Outputs (list): The output/classe images to transform.
				Channels (int): The image number of channels.
				ChannelFirst (bool): Is the color channel placed first in the images?
				InCropSizeX (int): The input images crop width.
				InCropSizeY (int): The input images crop height.
				OutCropSizeX (int): The output images crop width.
				OutCropSizeY (int): The output images crop height.
		"""
		rand = random.Random()
		
		SizeX, SizeY, channels = self._FindDimensions(Inputs[0], ChannelFirst)

		inputs  = []
		outputs = []
		count = 0
		while 1:
			randstate = rand.getstate()
		
			for i in range(len(Inputs)):
				rand.setstate(randstate)
				inputs.append(self.Transform(Inputs[i], rand, SizeX, SizeY, Channels, ChannelFirst, InCropSizeX, InCropSizeY, True))
	
			for i in range(len(Outputs)):
				rand.setstate(randstate)
				outputs.append(self.Transform(Outputs[i], rand, SizeX, SizeY, Channels, ChannelFirst, InCropSizeX, InCropSizeY, False))

			if InCropSizeX != OutCropSizeX or InCropSizeY != OutCropSizeY:
				for i in range(len(outputs)):
					outputs[i] = self.Crop(outputs[i], rand, InCropSizeX, InCropSizeY, OutCropSizeX, OutCropSizeY, 0, ChannelFirst)


			sum = 0.0
			for i in range(len(outputs)): # Test if an image is black.
				sum += outputs[i].sum()
			if 0.0 < sum or rand.uniform(0.0,1.0) < self.KeepEmptyOutput or (SafetyCount <= count and 0.0 < self.KeepEmptyOutput):
				for i in range(len(Inputs)):
					Inputs[i] = inputs[i]
				for i in range(len(Outputs)):
					Outputs[i] = outputs[i]
				return

			inputs.clear()
			outputs.clear()
			count += 1
			if SafetyCount < count:
				for i in range(len(Inputs)):
					ImagesIO.Write(Inputs[i], ChannelFirst, "Failed Inputs " + str(i) + ".png")
				for i in range(len(Outputs)):
					ImagesIO.Write(Outputs[i], ChannelFirst, "Failed Outputs " + str(i) + ".png")
				raise Exception("Failed to generate an image after " + str(self.SafetyCount) + " tries." +
								" Check output images, increase SafetyCount, or the variable MaxShiftRange." +
								" An image named FailedXXX.png was generated for debuging purposes.")







	def _FindDimensions(self, image, ChannelFirst: bool):
		""" This function finds the image dimensions.
			Args:
				image: The images to analyze.
				ChannelFirst (bool): Where is the channel encoded.
		"""
		shape = image.shape
		
		if len(shape) == 2: #Gray level image
			return shape[1], shape[0], 1
		
		if ChannelFirst == True:
			Channels = shape[0]
			Width = shape[2]
			Height = shape[1]
		else:
			Width = image.shape[1]
			Height = image.shape[0]
			Channels = image.shape[2]

		return Width, Height, Channels


	def _CheckDimensions(self, Inputs: list, Outputs: list, ChannelFirst: bool):
		""" This function checks that all images dimensions match.
			Args:
				Inputs (list) : The input images to analyze.
				Outputs (list): The output images to analyze.
				ChannelFirst (bool): Where is the channel encoded.
		"""
		shape = self._FindDimensions(Inputs[0], ChannelFirst)
		for i in range(1, len(Inputs)):
			if shape != self._FindDimensions(Inputs[i], ChannelFirst):
				raise Exception("Input images have different dimensions.")
		for i in range(len(Outputs)):
			if shape != self._FindDimensions(Outputs[i], ChannelFirst):
				raise Exception("Output/Input images have different dimensions.")
