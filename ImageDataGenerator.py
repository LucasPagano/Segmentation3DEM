import numpy as np
import os
import random
import sys
import threading
import warnings

import ImagesIO
import ImageTools
import Processing
import Tools
import Transformations

from torch.utils.data import Dataset, DataLoader, Sampler







class Generator(object):
	""" This class generates batches with real-time data augmentation.
		Args:
			OnTheFly (bool): Read the images on the fly (when needed)? If False, images are loaded first. Defaults to False.
			ChannelFirst (bool): Generate batch of images with the color channel first? Defaults to False.
			MaxLevel (float32): The maximum gray/color level value possible in the image. Defaults to 255.0.
				This value will be used for normalizations, add noise and modify the intensity.
			Width (int): The images width.
			Height (int): The images height.
			Channels (int): The images channels.
			Schuffle (bool): Schuffle images between each epoch? Defaults to False.
			MaxShiftRange (int): when a random crop is performed, it's the maximum distance the window can be cropped from the center. Defaults to 0.
			Rotate (bool): Perform (or not) a random rotation? Defaults to False.
			AngleRange (int): The rotation range. The random rotation angle can only be picked in [–AngleRange,AngleRange]. Defaults to 0.
			AngleStep (int): The rotation angle increment. Defaults to 1.
			RotateMode (str): The filling mode for empty pixels. Must be ‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’. Defaults to 'reflect'.
			FillingValues (list): The filling values to use if the rotation mode is 'constant'.
			Rotate90x (bool): Perform (or not) a random rotation by a multiple of 90 degrees? Defaults to False.
			Flip (bool): Randomly perform (or not) a flip/miror of the image along the X and/or Y axis?
			NoiseRange (int): The range is the number of pixels in which X% of the noise values will be. Defaults to 0, so no noise added.
			NoiseType  (str): The noise type (Uniform, Gaussian, Poisson). Defaults to Gaussian.
			NoisePercentage (int): The pixels percentage that will be modified by the noise? Defaults to 75.
			BrighterDarkerRange (int): The range is the maximum number of pixels to modify the images. Defaults to 0, so no modifications.
			BrighterDarkerType  (str): The enlightment variation type (Uniform, Gaussian,). Defaults to Uniform_Equal.
			KeepEmptyOutput (float): The probability to keep an empty/black output. Defaults to 1.
			Inputs (list): This list contains all the images or their paths (if OnTheFly==True) clustered by classes/directories.
			InputsSizes (list): The number of images per class.
			Outputs (list): This list contains all the output images or their paths (if OnTheFly==True), or will be empty for a classification task.
			Classification (bool): Is this a classfication task? Defaults to False.
			ClassificationClass0 (int): In case of classification task, it is the value to use when an element is negative for a class. This value
				is usually 0 or -1. Defaults to 0.
			InputSizeX (int): The input crop width.
			InputSizeY (int): The input crop height.
			OutputSizeX (int): The output crop width.
			OutputSizeY (int): The output crop height.
			nbCropPerImage (int): The number of crop to take per image. Defaults to 1.
		
		Examples:
			# Generation of segmentation batches
			import ImageDataGenerator
			import Processing
			from torch.utils.data import DataLoader
		
			generator = ImageDataGenerator.Generator(ChannelFirst=False)
			#generator.setShuffle(True) # Do not use with PyTorch, it can be performed by the DataLoader.
			generator.setFlip(True)
			generator.setRotate90x(True) # Or => generator.setRotate(True, 25, 1, 'constant', 255) # Only one rotation at a time.
			generator.setMaxShiftRange(100)
			#generator.setNoise(50, 'Uniform', 100) # Should not be used. It will generate warning.
			generator.setKeepEmptyOutputProbability(0.25)
			generator.LoadInputs("./DataSets/Test Segmentation Inputs/", OnTheFly=False, Classification=False)#, ClassificationClass0=-1.0)
			generator.LoadOutputs("./DataSets/Test Segmentation Outputs/")
			generator.setInputsDimensions(256, 256)
			generator.setOutputsDimensions(250, 250)
			BatchSize = 6
			#gen = generator.Keras(BatchSize, InputsNormalizer=Processing.Normalize, OutputsNormalizer=Processing.NormalizeBasic) # For Keras
			gen = generator.PyTorch(BatchSize, InputsNormalizer=Processing.Normalize, OutputsNormalizer=Processing.NormalizeBasic) # For PyTorch
			dl = DataLoader(gen, batch_size=BatchSize, shuffle=False, num_workers=4)
			device = 'cuda' if torch.cuda.is_available() else 'cpu'  # sets training to run on GPU if available
			for batch in enumerate(dl):
				input, output = batch['input'].to(device), batch['output'].to(device)
			
		
			# Generation of classification batches
			import ImageDataGenerator
			import Processing
		
			generator = ImageDataGenerator.Generator(ChannelFirst=False)
			generator.setShuffle(True)
			generator.setFlip(True)
			generator.setRotate(True, 25, 1, 'constant', 255) # Or => generator.setRotate90x(True) # Only one rotation at a time.
			generator.setMaxShiftRange(100)
			generator.LoadInputs("./DataSets/Test Classification Inputs/", OnTheFly=False, Classification=True, ClassificationClass0=-1.0)
			generator.setInputsDimensions(256, 256)
			BatchSize = 6
			gen = generator.Keras(BatchSize, InputsNormalizer=Processing.Normalize) # For Keras
			#gen = generator.PyTorch(BatchSize, InputsNormalizer=Processing.Normalize) # For PyTorch
			
			
			TODO:
				Add Affine/Elastic transformations
				Add the seed
	"""

	
	
	def __init__(self, ChannelFirst: bool=False):
		""" Simple initialization.
			Args:
				ChannelFirst (:obj:`bool`, optional): Return image batches with the color channel first? Defaults to False.
		"""
		self.__Reset__()
		self.ChannelFirst = ChannelFirst
	
	
	
	def __Reset__(self):
		self.OnTheFly = False
		self.ChannelFirst = False
		self.MaxLevel = 255.0
		self.Width = 0
		self.Height = 0
		self.Channels = 0
		self.Shuffle = False
		self.MaxShiftRange = 0
		self.Rotate90x = False
		self.Rotate = False
		self.AngleRange = 0
		self.AngleStep = 1
		self.RotateMode = 'reflect'
		self.FillingValues = None
		self.Flip = False
		self.NoiseRange = 0
		self.NoiseType = 'Gaussian'
		self.NoisePercentage = 75
		self.BrighterDarkerRange = 0
		self.BrighterDarkerType = 'Uniform_Equal'
		self.KeepEmptyOutput = 1.0
		
		self.Inputs = []
		self.InputSizes = []
		self.Outputs = []
		self.Classification = False
		self.ClassificationClass0 = 0
		self.InputSizeX = -1
		self.InputSizeY = -1
		self.OutputSizeX = -1
		self.OutputSizeY = -1
		self.OutputChannels = -1
		self.nbCropPerImage = 1






# ------------------------------------------------------------ Loaders ------------------------------------------------------------
	
	def LoadInputs(self, InputsDirPath: str, OnTheFly: bool=False, Classification: bool=False, ClassificationClass0: float=0.0):
		""" This function finds all the images in the given directory or in all the sub-directories. If it is a classification task the
				sub-directories must have the same number of images, else it does not matter and an over-sampling will be performed durnig batch
				generation.
			Args:
				InputsDirPath (str): The directory containing the input images.
				OnTheFly (bool, optional): If False then all the images are immedialely loaded, else they will be loaded on the fly when
					necesarry. Defaults to False.
				Classification (bool, optional): Is this generator used in a classification task? If False then all the sub-directories
					must have the same number of images. Defaults to False.
				ClassificationClass0 (int, optional): Which value set for the negative class. Defaults to 0.
		"""
		self.OnTheFly = OnTheFly
		self.Classification = Classification
		self.ClassificationClass0 = ClassificationClass0

	
		dirs = [dir for dir in os.listdir(InputsDirPath) if os.path.isdir(os.path.join(InputsDirPath, dir))]
		dirs.sort()
		print("\n%d directories / classes found in %s" % (len(dirs), InputsDirPath))
		
	
		if OnTheFly == True: # On The Fly
			print("Mode On The Fly activated => The images will be loaded only when necessary, except for one to determine dimensions.")
			if len(dirs) == 0: # No directories, so images should be in the given directory.
				if Classification == True:
					raise Exception("No directories/classes found for this classification task in " + InputsDirPath)
				images = ImagesIO.FindImages(InputsDirPath, NamesOnly=False, verbose=False)
				self.Inputs.append(images)
				self.InputSizes.append(len(images))
				print(" - %d image files in %s" % (len(images), InputsDirPath))
			
			else:
				for dir in dirs:
					path = os.path.join(InputsDirPath, dir)
					images = ImagesIO.FindImages(path, NamesOnly=False, verbose=False)
					self.Inputs.append(images)
					self.InputSizes.append(len(images))
					print(" - %d image files in %s" % (len(images), path))
			
			images = ImagesIO.LoadImagesList(self.Inputs[0][0:1], self.ChannelFirst, False, False)
			width, height, channels, first = ImageTools.Dimensions(images[0])
			self.Channels = channels
	
		else: # Direct/Immediate load
			if len(dirs) == 0: # No directories, so images should be in the given directory.
				if Classification == True:
					raise Exception("No directories/classes found for this classification task in " + InputsDirPath)
				images = ImagesIO.LoadImages(InputsDirPath, ChannelFirst=self.ChannelFirst, ReturnImagesList=False, verbose=True)
				self.Inputs.append(images)
				self.InputSizes.append(images.shape[0])
			else:
				for dir in dirs:
					images = ImagesIO.LoadImages(os.path.join(InputsDirPath, dir), ChannelFirst=self.ChannelFirst, ReturnImagesList=False, verbose=True)
					self.Inputs.append(images)
					self.InputSizes.append(images.shape[0])
				
				for input in self.Inputs: # Check the images dimensions in each directory.
					if input.shape[1:len(input.shape)] != self.Inputs[0].shape[1:len(input.shape)]:
						raise Exception("Images must have identical dimensions, even throughout different directories.")
				
			width, height, channels, first = ImageTools.Dimensions(self.Inputs[0][0])
			self.Channels = channels

		if Classification == False:
			for size in self.InputSizes: # Check the number of images in each directory/class
				if size != self.InputSizes[0]:
					raise Exception("Directories must contain the same number of (readable) images for non classicication tasks.")
		
		
		if len(self.Inputs) == 0: # No directories, so images should be in the given directory.
			raise Exception("No images found in %s", InputsDirPath)
		
		print("\n")
		sys.stdout.flush()



	def LoadOutputs(self, OutputsDirPath: str):
		""" This function finds all the images in the given directory or in all the sub-directories. The inputs must be given first, and the parameter
			Classification (optional in function LoadInputs) must be False.
			Args:
				OutputsDirPath (str): The directory containing the output images.
		"""
		if self.Classification == True:
			raise Exception("Output images cannot be given in a classification task.")
		
		dirs = [dir for dir in os.listdir(OutputsDirPath) if os.path.isdir(os.path.join(OutputsDirPath, dir))]
		dirs.sort()
		print("\n%d directories / classes found in %s" % (len(dirs), OutputsDirPath))
		
		if self.OnTheFly == True:
			if len(dirs) == 0: # No directories, so images should be in the given directory.
				images = ImagesIO.FindImages(OutputsDirPath, NamesOnly=False, verbose=False)
				self.Outputs.append(images)
				print(" - %d image files in %s" % (len(images), OutputsDirPath))
			else:
				for dir in dirs:
					path = os.path.join(OutputsDirPath, dir)
					images = ImagesIO.FindImages(path, NamesOnly=False, verbose=False)
					self.Outputs.append(images)
					print(" - %d image files in %s" % (len(images), path))
		else:
			if len(dirs) == 0: # No directories, so images should be in the given directory.
				self.Outputs.append(ImagesIO.LoadImages(OutputsDirPath, ChannelFirst=self.ChannelFirst, ReturnImagesList=False, verbose=True))
			else:
				for dir in dirs:
					self.Outputs.append(ImagesIO.LoadImages(os.path.join(OutputsDirPath, dir),
															ChannelFirst=self.ChannelFirst, ReturnImagesList=False, verbose=True))
				
				for out in self.Outputs:
					if out.shape[0] != self.InputSizes[0]:
						raise Exception("Directories must contain the same number of (readable) images in Input and Output.")

				for output in self.Outputs: # Check the images dimensions in each directory.
					if output.shape[1:len(output.shape)] != self.Outputs[0].shape[1:len(output.shape)]:
						raise Exception("Images must have identical dimensions, even through different directories.")

		if len(self.Outputs) == 0: # No directories, so images should be in the given directory.
			raise Exception("No images found in %s", OutputsDirPath)
		
		print("\n")
		sys.stdout.flush()

		if self.OnTheFly == True:
			images = ImagesIO.LoadImagesList(self.Outputs[0][0:1], self.ChannelFirst, False, False)
		else:
			images = self.Outputs[0]
		width, height, channels, first = ImageTools.Dimensions(images[0])
		self.OutputChannels = channels


	def __iter__(self): # Useless because the initialization is performed into the function init.
		return self

























# ------------------------------------------------------------ Setters ------------------------------------------------------------
	def setMaxShiftRange(self, MaxShiftRange: int):
		""" This function sets the maximum shift range: when a crop is performed, it's the maximum distance the window can be cropped from the center
			of the image. This value must be positive, but there is no maximum limit as the software limits itself to the image dimensions.
			Args:
				MaxShiftRange (int): the maximum value (must be positive). Defaults to 0.
		"""
		if MaxShiftRange < 0:
			raise Exception("MaxShiftRange < 0")
		self.MaxShiftRange = MaxShiftRange


	def setRotate(self, Rotation: bool, AngleRange: int, AngleStep: int=1, Mode: str='reflect', FillingValues: list=None):
		""" This function sets the rotation parameters.
			Args:
				Rotation (bool): Perform (or not) a rotation? Defaults to False.
				MaxAngleRange (int): The rotation range. The random rotation angle can only be picked in [–AngleRange,AngleRange]. Defaults to 0.
				AngleStep (int): The rotation angle increment. Defaults to 1.
				RotateMode (str): The filling mode for empty pixels. Must be ‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’. Defaults to 'reflect'.
				FillingValues (list): The filling values to use if the rotation mode is 'constant'.
		"""
		if self.Rotate90x == True:
			raise Exception("Only one rotation at a time. Set Rotate90x to False to use Rotate.")
		
		if Mode not in {'constant', 'nearest', 'reflect', 'wrap'}:
			raise Exception("Mode not in ('constant', 'nearest', 'reflect', 'wrap').")
		
		self.Rotate = Rotation
		self.AngleRange = AngleRange
		self.AngleStep = AngleStep
		self.RotateMode = Mode
		self.FillingValues = FillingValues
	
	
	def setRotate90x(self, Rotation90x: bool):
		""" This function sets the rotation by a multiple of 90 degrees variable.
			Args:
				Rotation90x (bool): Perform (or not) a rotation by a multiple of 90 degrees? Defaults to False.
		"""
		if self.Rotate == True:
			raise Exception("Only one rotation at a time. Set Rotate to False to use Rotate90x.")
		
		self.Rotate90x = Rotation90x


	def setFlip(self, Flip: bool):
		""" This function sets the flip (along X and Y axes) variable.
			Args:
				Flip (bool): Perform (or not) image flipping along X and Y axes. Defaults to False.
		"""
		self.Flip = Flip


	def setNoise(self, Range: int, Type: str, Percentage: int):
		""" This function sets the noise parameters. More parameters are available in Transformations to fine tune Gaussian and Poisson noises.
			Args:
				Range (int): The range is the number of pixels in which X% of the noise values will be. Defaults to 0, so no noise added.
				Type  (str): The noise type (Uniform, Gaussian, Poisson). Defaults to Gaussian.
				Percentage (int): The pixels percentage that will be modified by the noise. Defaults to 75.
		"""
		warnings.warn("\nWarning: Do not add noise using this function. Develop a dedicated layer!!!\n")
		
		if Range < 0:
			raise Exception("Range < 0")
		if Type not in ['Uniform', 'Gaussian', 'Poisson']:
			raise Exception("Unsupported noise type.")
		if Percentage < 0 or 100 < Percentage:
			raise Exception("Percentage < 0 or 100 < Percentage")
		self.NoiseRange = Range
		self.NoiseType = Type
		self.NoisePercentage = Percentage


	def setBrighterDarker(self, Range: int, Type: str):
		""" This function sets the enlightment variations parameters.
			Args:
				Range (int): The range is the maximum number of pixels to modify the images. Defaults to 0, so no modifications.
				Type  (str): The enlightment variation type (Uniform, Gaussian,). Defaults to Uniform_Equal.
		"""
		if Range < 0:
			raise Exception("Range < 0")
		if Type not in ['Uniform_Equal', 'Uniform_PerChannel', 'Gaussian_Equal', 'Gaussian_PerChannel']:
			raise Exception("Unsupported type.")
		self.BrighterDarkerRange = Range
		self.BrighterDarkerType = Type

		
	def setKeepEmptyOutputProbability(self, Probability: float):
		""" This function sets the probability to keep a generate empty/black output.
			Args:
				Probability (float): The probability to keep an empty output.
		"""
		if Probability < 0.0 or 1.0 < Probability:
			raise Exception("This is a probability, so it must be in range [0.0, 1.0].")
		
		self.KeepEmptyOutput = Probability


	def setShuffle(self, Shuffle: bool):
		""" This function sets the flip (along X and Y axes) variable. Useless with PyTorch, it is performed by the DataLoader.
			Args:
				Shuffle (bool): Perform (or not) Schuffling between each epoch? Defaults to False.
		"""
		self.Shuffle = Shuffle


	def setInputsDimensions(self, SizeX: int, SizeY: int):
		""" This function sets the input crop dimensions.
			Args:
				SizeX (int): The crop X size.
				SizeY (int): The crop Y size.
		"""
		if SizeX <= 0 or SizeY <= 0:
			raise Exception("Dimensions must be positive")
		self.InputSizeX = SizeX
		self.InputSizeY = SizeY


	def setOutputsDimensions(self, SizeX: int, SizeY: int):
		""" This function sets the output crop dimensions. Useless for a classification task.
			Args:
				SizeX (int): The crop X size.
				SizeY (int): The crop Y size.
		"""
		if self.InputSizeX <= 0 or self.InputSizeY <= 0:
			raise Exception("The input dimensions must be given first.")
		if SizeX <= 0 or SizeY <= 0:
			raise Exception("Dimensions must be positive")
		if self.InputSizeX < SizeX or self.InputSizeY < SizeY:
			raise Exception("The output dimensions must be smaller or equal to the input dimensions.")
		self.OutputSizeX = SizeX
		self.OutputSizeY = SizeY
	
	
	def setMaxLevel(self, Max: float):
		""" This function sets the maximum gray/color level value possible in the input images. This value with be used for normalizations, add
				noise and modify the intensity.
			Args:
				Max (float): The maximum possible value. Defaults to 255.
		"""
		if Max <= 0.0:
			raise Exception("Maximum level <= 0.0")
		self.MaxLevel = Max

	
	def setCropPerImage(self, nbCropPerImage: int):
		""" This function sets the number of crop to cut per image.
			Args:
				nbCropPerImage (int): The number of crops.
		"""
		if nbCropPerImage <= 0:
			raise Exception("The number of crop per image must be at least 1.")
		self.nbCropPerImage = nbCropPerImage































# ------------------------------------------------------------ Checks ------------------------------------------------------------

	def _BasicChecks(self, BatchSize: int, InputsNormalizers, OutputsNormalizers):
		""" This function performs basic checks.
			Args:
				BatchSize (int): The number of inputs in the batch.
				InputsNormalizer (object):
				OutputsNormalizers (list):
		"""
		
		if self.InputSizeX < 1 or self.InputSizeY < 1:
			raise Exception("Input dimensions / crops must be given beforehand.")
		
		if InputsNormalizers != None:
			if len(InputsNormalizers) != len(self.Inputs):
				raise Exception("len(InputsNormalizers) != len(Inputs), %d vs %d" % (len(InputsNormalizers), len(self.Inputs)))
			
			if self.OnTheFly == False:
				for input, normalizer in zip(self.Inputs,InputsNormalizers):
					normalizer.Normalize(input)
			
			if self.FillingValues is not None:
				values = []
				for normalizer, value in zip(InputsNormalizers, self.FillingValues):
					values.append(normalizer.NormalizeScalar(value))
				self.FillingValues = values
		
		if self.Classification == True:
			self._ClassificationChecks(BatchSize)
		else:
			self._SegmentationChecks(BatchSize, OutputsNormalizers)

		if 1 < self.nbCropPerImage and self.OnTheFly == True:
			warnings.warn("Code not optimized (yet) for multiple crops per image with on the fly mode.")
	
	
	
	def _ClassificationChecks(self, BatchSize: int):
		""" This function performs basic checks for the classification mode.
			Args:
				BatchSize (int): The number of inputs in the batch.
		"""
		
		nbClasses = len(self.Inputs)
		
		if BatchSize % nbClasses != 0:
			raise Exception("The batch size must be a multiple of the classes number.")
		
		if min(self.InputSizes)*self.nbCropPerImage < BatchSize / nbClasses:
			raise Exception("The batch size is too big in comparison of the images number in the smallest class.")



	def _SegmentationChecks(self, BatchSize: int, OutputsNormalizers):
		""" This function performs basic checks for the segmentation (non classification) mode.
			Args:
				BatchSize (int): The number of inputs in the batch.
				OutputsNormalizers (list): The preprocessing (data normalization) to apply to each output.
		"""
		
		if len(self.Inputs) == 0 :
			raise Exception("No input images loaded.")
		
		if len(self.Outputs) == 0 :
			raise Exception("No output images loaded.")
		
		nbClasses = len(self.Inputs)
		
		if BatchSize <= 0:
			raise Exception("The batch size must be positive.")
		
		if self.InputSizes[0]*self.nbCropPerImage < BatchSize:
			raise Exception("The batch size is too big in comparison of the images number.")
		
		if self.OutputSizeX < 1 or self.OutputSizeY < 1:
			raise Exception("Output dimensions / crops must be given beforehand for non classification tasks.")

		if OutputsNormalizers != None and self.OnTheFly == False:
			for out, norm in zip(self.Outputs, OutputsNormalizers):
				norm.Normalize(out)























# ------------------------------------------------------------ Keras Generator ------------------------------------------------------------

	def Keras(self, BatchSize: int, InputsNormalizers=None, OutputsNormalizers: list=None):
		""" This function creates a Keras compatible data generator.
			Args:
				BatchSize (int): The number of inputs in the batch.
				InputsNormalizer (list, optional): The preprocessing (data normalization) to apply to each inputs.
				OutputsNormalizers (list, optional): The preprocessing (data normalization) to apply to each output.
		"""
		
		self._BasicChecks(BatchSize, InputsNormalizers, OutputsNormalizers)
		
		if self.Classification == True:
			return _KerasClassification(self, BatchSize, InputsNormalizers)
		else:
			return _KerasSegmentation(self, BatchSize, InputsNormalizers, OutputsNormalizers)







# ------------------------------------------------------------ PyTorch Generator ------------------------------------------------------------
	
	def PyTorchDataset(self, BatchSize: int, InputsNormalizers=None, OutputsNormalizers=None):
		""" This function creates a PyTorch dataset.
			Args:
				BatchSize (int): The number of inputs in the batch.
				InputsNormalizer (:obj:, optional): The preprocessing (data normalization) to apply to the inputs.
					One normalizer is applied per input, so the number of normalizers must match the number of inputs.
				OutputsNormalizers (list, optional): The preprocessing (data normalization) to apply to the outputs.
					One normalizer is applied per output, so the number of normalizers must match the number of outputs.
		"""
		self._BasicChecks(BatchSize, InputsNormalizers, OutputsNormalizers)
		
		if self.Shuffle == True:
			raise Exception("Do not use shuffling with PyTorch, it can be performed by the DataLoader.")

		if self.Classification == True:
			raise Error("Classification with PyTorch not implemented (yet)!")
			#return _PyTorchClassification(self, BatchSize, InputsNormalizer)
		else:
			#return _PyTorchSegmentationOld(self, BatchSize, InputsNormalizers, OutputsNormalizers)
			return _PyTorchSegmentation(self, BatchSize, InputsNormalizers, OutputsNormalizers)


	def PyTorch(self, BatchSize: int, InputsNormalizers=None, OutputsNormalizers: list=None, Workers: int=5):
		""" This function creates a PyTorch compatible data generator, so a DataLoader.
			Args:
				BatchSize (int): The number of inputs in the batch.
				InputsNormalizers (:obj:, optional): The preprocessing (data normalization) to apply to the inputs.
					One normalizer is applied per input, so the number of normalizers must match the number of inputs.
				OutputsNormalizers (list, optional): The preprocessing (data normalization) to apply to the outputs.
					One normalizer is applied per output, so the number of normalizers must match the number of outputs.
				Workers (int, optional): The number of workers, so the number of images to process in parallel.
		"""
		shuffle = self.Shuffle
		self.Shuffle = False
		dataset = self.PyTorchDataset(BatchSize, InputsNormalizers, OutputsNormalizers)
		self.Shuffle = shuffle
		
		
		if self.Classification == True:
			raise Error("Classification with PyTorch not done (yet)!")
		else:
			sampler = None
			"""
			if self.nbCropPerImage == 1: # Older version
				sampler = None
			else:
				sampler = SegmentationSampler(self, BatchSize)
				shuffle = False # TODO Find a way to remove it!!!
			"""
		if self.Shuffle == True: # Workaround because otherwise the first batch is not shuffled.
			Tools.ShuffleUnisonXY(self.Inputs, self.Outputs, False)
		
		return DataLoader(dataset, batch_size=BatchSize, shuffle=shuffle, num_workers=Workers, sampler=sampler)




























# ------------------------------------------------------------ Iterators ------------------------------------------------------------

class _SingleIterator(object):
	
	def __init__(self, CallerClass, BatchSize: int, Iterator: bool=False):
		self.CallerClass = CallerClass
		
		self.BatchSize = BatchSize
		
		self.Iterator = Iterator
		
		self.Reset()
		
		self.lock = threading.Lock()
		self.IndexGenerator = self.FlowIndex()
	
	
	def Reset(self):
		self.Index = 0
		self.nbCrops = 0
	
	
	def FlowIndex(self):
		self.Reset()
		
		while 1:
			if self.CallerClass.InputSizes[0] == self.Index:
				self.Index = 0
				if self.Iterator == True:
					raise StopIteration
		
			yield self.Index, self.nbCrops
			
			self.nbCrops += 1
			if self.nbCrops == self.CallerClass.nbCropPerImage:
				self.Index += 1
				self.nbCrops = 0


	def __iter__(self):
		return self # needed if we want to do something like: for x, y in data_gen.flow(...):
	
	
	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)





class _MultipleIterator(object):
	
	def __init__(self, CallerClass, BatchSize, Iterator: bool=False):
		self.CallerClass = CallerClass
		
		self.Iterator = Iterator
		
		self.nbClasses = len(CallerClass.Inputs)
		self.BatchSize = BatchSize
		self.FractionBatchSize = int(BatchSize / self.nbClasses)
		
		self.Indexes = np.ndarray(shape=(self.nbClasses), dtype=np.int32)
		self.Indexes.fill(0)
		self.nbCrops = 0
		
		self.lock = threading.Lock()
		self.IndexesGenerator = self.FlowIndexes()
	
	
	def Reset(self):
		self.Indexes.fill(0)
		self.nbCrops = 0
	
	
	def FlowIndexes(self):
		self.Reset()
		
		while 1:
			stop = False
			for i in range(self.nbClasses):
				if self.Indexes[i] == self.CallerClass.InputSizes[i]:
					self.Indexes[i] = 0
					if self.Iterator and self.CallerClass.InputSizes[i] == max(self.CallerClass.InputSizes[i]):
						stop = True
			if stop:
				raise StopIteration
		
		
			yield self.Indexes, self.nbCrops
			
			self.nbCrops += 1
			if self.nbCrops == self.CallerClass.nbCropPerImage:
				self.Indexes += 1
				self.nbCrops = 0


	def __iter__(self):
		return self # needed if we want to do something like: for x, y in data_gen.flow(...):
	
	
	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)
	
	
	










	
	
	
	
	
	
	
	
	
	
	
	














# ------------------------------------------------------------ Keras Classes ------------------------------------------------------------

class _KerasClassification(_MultipleIterator):
	
	def __init__(self, CallerClass, BatchSize, Normalizer):
		
		super().__init__(CallerClass, BatchSize)
		
		self.Normalizer = Normalizer
		
		self.EpochOver = False
		self.TotalBatches = 0
		self.nbBatch = 0
		
		self.nbBatchPerEpoch = int(max(CallerClass.InputSizes)*CallerClass.nbCropPerImage/self.FractionBatchSize)
		self.SamplePerEpoch = self.nbBatchPerEpoch * BatchSize
		
		self.Y = np.ndarray(shape=(BatchSize, self.nbClasses), dtype=np.float32)
		self.Y.fill(CallerClass.ClassificationClass0)
		for i in range(0, BatchSize):
			self.Y[i][i%self.nbClasses] = 1.0

		self.transformations = Transformations.Transformations(self.CallerClass.MaxShiftRange, self.CallerClass.Flip, self.CallerClass.Rotate90x,
																self.CallerClass.Rotate, self.CallerClass.AngleRange, self.CallerClass.AngleStep,
																self.CallerClass.RotateMode, self.CallerClass.FillingValues,
																self.CallerClass.BrighterDarkerRange, self.CallerClass.BrighterDarkerType,
																self.CallerClass.KeepEmptyOutput, Normalizer)

		self.GlobalIndex = 0
	


	def next(self):
		
		if self.CallerClass.ChannelFirst == True:
			X = np.ndarray(shape=(self.BatchSize, self.CallerClass.Channels, self.CallerClass.InputSizeY, self.CallerClass.InputSizeX), dtype=np.float32)
		else:
			X = np.ndarray(shape=(self.BatchSize, self.CallerClass.InputSizeY, self.CallerClass.InputSizeX, self.CallerClass.Channels), dtype=np.float32)
	
		pos = 0
		for ind in range(0, self.FractionBatchSize):
			with self.lock:
				Indexes, Crops = next(self.IndexesGenerator)
			
				if self.CallerClass.Shuffle == True and Crops  == 0:
					for c in range(self.nbClasses): # Shuffle when necessary.
						if Indexes[c] == 0:
							Tools.Shuffle(self.CallerClass.Inputs[c])
		
			for c in range(self.nbClasses):
				if self.CallerClass.OnTheFly == True:
					array = ImagesIO.LoadImagesList([self.CallerClass.Inputs[c][Indexes[c]]], self.CallerClass.ChannelFirst, False, False)[0]
					if self.Normalizer is not None:
						self.Normalizer.Normalize(array)
				else:
					array = self.CallerClass.Inputs[c][Indexes[c]]
				
				X[pos] = self.transformations.TransformSingle(array, self.CallerClass.Channels, self.CallerClass.ChannelFirst,
															self.CallerClass.InputSizeX, self.CallerClass.InputSizeY, True)
				pos += 1
		
		if 0 < self.CallerClass.NoiseRange:
			self.transformations.Noise(X, self.CallerClass.NoiseRange, self.CallerClass.NoiseType, self.CallerClass.NoisePercentage)

		self.TotalBatches += 1 # Update various counts
		self.nbBatch += 1
		#if self.nbBatchPerEpoch <= self.nbBatch:
			#self.EpochOver = True
		return X, self.Y
	
	
	"""
	def isEpochOver(self):
		return self.EpochOver
	
	def NewEpoch(self):
		self.Reset()
		self.nbBatch = 0
		self.EpochOver = False
	
	def FullReset(self):
		self.TotalBatches = 0
		self.NewEpoch()
	"""
	def getTotalBatches(self):
		return self.TotalBatches
	
	def getNbBatchPerEpoch(self):
		return self.nbBatchPerEpoch
	
	def getStepsPerEpoch(self):
		return self.nbBatchPerEpoch
	
	def getSamplePerEpoch(self):
		return self.SamplePerEpoch
















class _KerasSegmentation(_SingleIterator):
	
	def __init__(self, CallerClass, BatchSize, InputNormalizers, OutputNormalizers):
		
		super().__init__(CallerClass, BatchSize)
		
		self.InputNormalizers  = InputNormalizers
		self.OutputNormalizers = OutputNormalizers
		
		self.nbInputs = len(CallerClass.Inputs)
		self.nbClasses = len(CallerClass.Outputs)
		
		self.EpochOver = False
		self.TotalBatches = 0
		self.nbBatch = 0
		
		self.nbBatchPerEpoch = int(CallerClass.InputSizes[0]*CallerClass.nbCropPerImage/BatchSize)
		self.SamplePerEpoch = self.nbBatchPerEpoch * BatchSize
	
		self.transformations = Transformations.Transformations(self.CallerClass.MaxShiftRange, self.CallerClass.Flip, self.CallerClass.Rotate90x,
														   self.CallerClass.Rotate, self.CallerClass.AngleRange, self.CallerClass.AngleStep,
														   self.CallerClass.RotateMode, self.CallerClass.FillingValues,
														   self.CallerClass.BrighterDarkerRange, self.CallerClass.BrighterDarkerType,
														   self.CallerClass.KeepEmptyOutput, InputNormalizers)




	def next(self):

		Xs = []
		if self.CallerClass.ChannelFirst == True:
			for i in range(self.nbInputs):
				Xs.append(np.ndarray(shape=(self.BatchSize, self.CallerClass.Channels, self.CallerClass.InputSizeY, self.CallerClass.InputSizeX),
									 dtype=np.float32))
		else:
			for i in range(self.nbInputs):
				Xs.append(np.ndarray(shape=(self.BatchSize, self.CallerClass.InputSizeY, self.CallerClass.InputSizeX, self.CallerClass.Channels),
									 dtype=np.float32))

		Ys = []
		if self.CallerClass.ChannelFirst == True:
			for i in range(self.nbClasses):
				Ys.append(np.ndarray(shape=(self.BatchSize, self.CallerClass.OutputChannels, self.CallerClass.OutputSizeY, self.CallerClass.OutputSizeX),
									 dtype=np.float32))
		else:
			for i in range(self.nbClasses):
				Ys.append(np.ndarray(shape=(self.BatchSize, self.CallerClass.OutputSizeY, self.CallerClass.OutputSizeX, self.CallerClass.OutputChannels),
									 		dtype=np.float32))

		for ind in range(0, self.BatchSize):
			inputs = []
			outputs = []
			
			with self.lock:
				Index, Crop = next(self.IndexGenerator)
			
				if self.CallerClass.Shuffle == True and Index == 0 and Crop == 0: # Shuffle when necessary.
					Tools.ShuffleUnisonXY(self.CallerClass.Inputs, self.CallerClass.Outputs, False)
		
			if self.CallerClass.OnTheFly == True:
				for c in range(self.nbInputs):
					array = ImagesIO.LoadImagesList([self.CallerClass.Inputs[c][Index]], self.CallerClass.ChannelFirst, False, False)[0]
					if self.InputNormalizers[c] is not None:
						self.InputNormalizers[c].Normalize(array)
					inputs.append(array)
			
				for c in range(self.nbClasses):
					array = ImagesIO.LoadImagesList([self.CallerClass.Outputs[c][Index]], self.CallerClass.ChannelFirst, False, False)[0]
					if self.OutputNormalizers is not None:
						raise Exception("To Check!")
						for norm, o in zip(self.OutputNormalizers, range(len(self.OutputNormalizers))):
							norm.Normalize(array[o])
					outputs.append(array)
			else:
				for c in range(self.nbInputs):
					inputs.append(self.CallerClass.Inputs[c][Index])
			
				for c in range(self.nbClasses):
					outputs.append(self.CallerClass.Outputs[c][Index])
			
			self.transformations.TransformUnison(inputs, outputs, self.CallerClass.Channels, self.CallerClass.ChannelFirst,
								self.CallerClass.InputSizeX, self.CallerClass.InputSizeY, self.CallerClass.OutputSizeX, self.CallerClass.OutputSizeY)
			
			for c in range(self.nbInputs):
				Xs[c][ind] = inputs[c]
			
			for c in range(self.nbClasses):
				Ys[c][ind] = outputs[c]
			
		
		if 0 < self.CallerClass.NoiseRange:
			for i in range(self.nbInputs):
				self.transformations.Noise(Xs[i], self.CallerClass.NoiseRange, self.CallerClass.NoiseType, self.CallerClass.NoisePercentage)

		self.TotalBatches += 1 # Update various counts
		self.nbBatch += 1
		#if self.nbBatchPerEpoch <= self.nbBatch: # Bug compatibility, do not use until fixed.
		#	self.EpochOver = True
		if len(Xs) == 1:
			return Xs[0], Ys[0]
		return Xs, Ys
	
	
	""" # Bug compatibility, do not use until fixed.
	def isEpochOver(self):
		return self.EpochOver
	
	def NewEpoch(self):
		self.Reset()
		self.nbBatch = 0
		self.EpochOver = False
	
	def FullReset(self):
		self.TotalBatches = 0
		self.NewEpoch()
	"""
	
	def getTotalBatches(self):
		return self.TotalBatches
	
	def getNbBatchPerEpoch(self):
		return self.nbBatchPerEpoch
	
	def getStepsPerEpoch(self):
		return self.nbBatchPerEpoch
	
	def getSamplePerEpoch(self):
		return self.SamplePerEpoch





















# ------------------------------------------------------------ PyTorch Classes ------------------------------------------------------------

"""
class _PyTorchClassification(Dataset):
	
	def __init__(self, CallerClass, BatchSize, Normalizer):
		
		super().__init__()
		
		self.CallerClass = CallerClass
		self.Normalizer = Normalizer
		
		self.nbClasses = len(CallerClass.Inputs)
		self.BatchSize = BatchSize
		self.FractionBatchSize = int(BatchSize / self.nbClasses)
		
		self.Indexes = np.ndarray(shape=(self.nbClasses), dtype=np.int32)
		self.Indexes.fill(0)
		
		self.EpochOver = False
		self.TotalBatches = 0
		self.nbBatch = 0
		
		self.nbBatchPerEpoch = int(max(CallerClass.InputSizes)/self.FractionBatchSize)
		self.SamplePerEpoch = self.nbBatchPerEpoch * BatchSize
		
		self.Y = np.ndarray(shape=(BatchSize, self.nbClasses), dtype=np.float32)
		self.Y.fill(CallerClass.ClassificationClass0)
		for i in range(0, BatchSize):
			self.Y[i][i%self.nbClasses] = 1.0



	def NextIndexes(self):
		indexes = np.copy(self.Indexes)
		self.Indexes += self.FractionBatchSize
		for i in range(self.nbClasses):
			if self.CallerClass.InputSizes[i] < self.Indexes[i]+self.FractionBatchSize:
				self.Indexes[i] = 0
		return indexes
	
	
	
	def __getitem__(self, index):
		
		Indexes = self.NextIndexes()
		
		if self.CallerClass.Shuffle == True:
			for i in range(self.nbClasses): # Shuffle when necessary.
				if Indexes[i] == 0:
					Tools.Shuffle(self.CallerClass.Inputs[i])
		
		self.TotalBatches += 1 # Update various counts
		self.nbBatch += 1
		if self.nbBatchPerEpoch <= self.nbBatch:
			self.EpochOver = True
		
		if self.CallerClass.ChannelFirst == True:
			X = np.ndarray(shape=(self.BatchSize, self.CallerClass.Channels, self.CallerClass.InputSizeY, self.CallerClass.InputSizeX), dtype=np.float32)
		else:
			X = np.ndarray(shape=(self.BatchSize, self.CallerClass.InputSizeY, self.CallerClass.InputSizeX, self.CallerClass.Channels), dtype=np.float32)
		
		
		for c in range(self.nbClasses):
			if self.CallerClass.OnTheFly == True:
				array = ImagesIO.LoadImagesList(self.CallerClass.Inputs[c][Indexes[c]:Indexes[c]+self.FractionBatchSize],
												self.CallerClass.ChannelFirst, False, False)
				#self.CallerClass._CheckDimensions(array)
				if self.Normalizer is not None:
					self.Normalizer(array)
				start = 0
				end = self.FractionBatchSize
			else:
				array = self.CallerClass.Inputs[c]
				start = Indexes[c]
				end = Indexes[c]+self.FractionBatchSize

			i = c
			for ind in range(start, end):
				X[i] = bt.RandomTransform(array[ind],
										  self.CallerClass.Width, self.CallerClass.Height, self.CallerClass.Channels, self.CallerClass.ChannelFirst,
										  self.CallerClass.InputSizeX, self.CallerClass.InputSizeY, self.CallerClass.MaxShiftRange,
										  self.CallerClass.Flip, self.CallerClass.Rotate90x,
										  self.CallerClass.Rotate, self.CallerClass.AngleRange, self.CallerClass.AngleStep,
										  self.CallerClass.RotateMode, self.CallerClass.FillingValues,
										  self.CallerClass.BrighterDarkerRange, self.CallerClass.BrighterDarkerType,
										  self.Normalizer)
				i += self.nbClasses

		if 0 < self.CallerClass.NoiseRange:
			bn.RandomNoise(X, self.CallerClass.NoiseType, self.CallerClass.NoiseRange, self.CallerClass.NoisePercentage)

		return X, self.Y
	
	
	
	
	
	def __len__(self):
		return self.getNbBatchPerEpoch()
	
	def isEpochOver(self):
		return self.EpochOver
	
	def NewEpoch(self):
		self.Indexes.fill(0)
		self.nbBatch = 0
		self.EpochOver = False
	
	def getTotalBatches(self):
		return self.TotalBatches
	
	def FullReset(self):
		self.TotalBatches = 0
		self.NewEpoch()
	
	def getNbBatchPerEpoch(self):
		return self.nbBatchPerEpoch
	
	def getStepsPerEpoch(self):
		return self.nbBatchPerEpoch
	
	def getSamplePerEpoch(self):
		return self.SamplePerEpoch

"""






class _PyTorch_SingleIterator(_SingleIterator):
	
	def __init__(self, CallerClass, BatchSize):
		super().__init__(CallerClass, BatchSize, Iterator=True)

	def next(self):
		with self.lock:
			Index, Crop = next(self.IndexGenerator)
		
			if Index == 0 and Crop == 0 and self.CallerClass.Shuffle == True:
				Tools.ShuffleUnisonXY(self.CallerClass.Inputs, self.CallerClass.Outputs, False)

		return Index



class SegmentationSampler(Sampler):
	""" Samples elements sequentially, always in the same order.
		Args:
			CallerClass : The calling class.
			BatchSize (int): The batch size.
	"""
	
	def __init__(self, CallerClass, BatchSize):
		self.CallerClass = CallerClass
		self.BatchSize = BatchSize
	
	def __iter__(self):
		return _PyTorch_SingleIterator(self.CallerClass, self.BatchSize)
	
	def __len__(self):
		return self.CallerClass.InputSizes[0] * self.CallerClass.nbCropPerImage



class _PyTorchSegmentationOld(Dataset):
	
	def __init__(self, CallerClass, BatchSize, InputNormalizers, OutputNormalizers):
		
		super().__init__()
		
		self.CallerClass = CallerClass
		self.BatchSize = BatchSize
		self.InputNormalizers = InputNormalizers
		self.OutputNormalizers = OutputNormalizers
		
		self.nbBatchPerEpoch = int(CallerClass.InputSizes[0]*CallerClass.nbCropPerImage/BatchSize)
		
		self.nbInputs = len(CallerClass.Inputs)
		self.nbClasses = len(CallerClass.Outputs)
	
		self.transformations = Transformations.Transformations(self.CallerClass.MaxShiftRange, self.CallerClass.Flip,
															self.CallerClass.Rotate90x, self.CallerClass.Rotate,
															self.CallerClass.AngleRange, self.CallerClass.AngleStep,
															self.CallerClass.RotateMode, self.CallerClass.FillingValues,
															self.CallerClass.BrighterDarkerRange, self.CallerClass.BrighterDarkerType,
															self.CallerClass.KeepEmptyOutput, InputNormalizers)
	
	
	
	
	def __getitem__(self, index):
		
		inputs = []
		outputs = []
		
		if self.CallerClass.OnTheFly == True:
			for c in range(self.nbInputs):
				image = ImagesIO.LoadImagesList([self.CallerClass.Inputs[c][index]], self.CallerClass.ChannelFirst, False, False)[0]
				if self.InputNormalizers is not None:
					self.InputNormalizers[c].Normalize(image)
				inputs.append(image)
			
			for c, norm in zip(range(self.nbClasses), self.OutputNormalizers):
				image = ImagesIO.LoadImagesList([self.CallerClass.Outputs[c][index]], self.CallerClass.ChannelFirst, False, False)[0]
				norm.Normalize(image)
				outputs.append(image)
		else:
			for c in range(self.nbInputs):
				inputs.append(self.CallerClass.Inputs[c][index])
			
			for c in range(self.nbClasses):
				outputs.append(self.CallerClass.Outputs[c][index])
		
		self.transformations.TransformUnison(inputs, outputs, self.CallerClass.Channels, self.CallerClass.ChannelFirst,
								self.CallerClass.InputSizeX, self.CallerClass.InputSizeY, self.CallerClass.OutputSizeX, self.CallerClass.OutputSizeY)

		if 0 < self.CallerClass.NoiseRange:
			for input, normalizer in zip(inputs, self.InputNormalizers):
				self.transformations.Noise(input, self.CallerClass.NoiseRange, self.CallerClass.NoiseType, self.CallerClass.NoisePercentage, normalizer)
		
		if len(inputs) == 1:
			X = inputs[0].copy()
		else:
			X = np.stack(inputs, axis=0)
		
		if len(outputs) == 1:
			Y = outputs[0].copy()
		else:
			Y = np.stack(outputs, axis=0)
		
		return {'input': X, 'output': Y}





	def __len__(self):
		return self.CallerClass.InputSizes[0] * self.CallerClass.nbCropPerImage

	def getNbBatchPerEpoch(self):
		return self.nbBatchPerEpoch







class _PyTorchSegmentation(Dataset):
	
	def __init__(self, CallerClass, BatchSize, InputNormalizers, OutputNormalizers):
		
		super().__init__()
		
		self.CallerClass = CallerClass
		self.InputNormalizers = InputNormalizers
		self.OutputNormalizers = OutputNormalizers
		
		self.nbBatchPerEpoch = int(CallerClass.InputSizes[0]*CallerClass.nbCropPerImage/BatchSize)
		self.Length = CallerClass.InputSizes[0] * CallerClass.nbCropPerImage
		
		self.nbInputs = len(CallerClass.Inputs)
		self.nbClasses = len(CallerClass.Outputs)
	
		self.transformations = Transformations.Transformations(self.CallerClass.MaxShiftRange, self.CallerClass.Flip,
															self.CallerClass.Rotate90x, self.CallerClass.Rotate,
															self.CallerClass.AngleRange, self.CallerClass.AngleStep,
															self.CallerClass.RotateMode, self.CallerClass.FillingValues,
															self.CallerClass.BrighterDarkerRange, self.CallerClass.BrighterDarkerType,
															self.CallerClass.KeepEmptyOutput, InputNormalizers)
		
	
	
	
	def __getitem__(self, index):
		
		index = int(index / self.CallerClass.nbCropPerImage) # Convert index into the real index because of the number of crops per image.
		inputs = []
		outputs = []
		
		if self.CallerClass.OnTheFly == True:
			for c in range(self.nbInputs):
				image = ImagesIO.LoadImagesList([self.CallerClass.Inputs[c][index]], self.CallerClass.ChannelFirst, False, False)[0]
				if self.InputNormalizers is not None:
					self.InputNormalizers[c].Normalize(image)
				inputs.append(image)
			
			for c, norm in zip(range(self.nbClasses), self.OutputNormalizers):
				image = ImagesIO.LoadImagesList([self.CallerClass.Outputs[c][index]], self.CallerClass.ChannelFirst, False, False)[0]
				norm.Normalize(image)
				outputs.append(image)
		else:
			for c in range(self.nbInputs):
				inputs.append(self.CallerClass.Inputs[c][index])
			
			for c in range(self.nbClasses):
				outputs.append(self.CallerClass.Outputs[c][index])
		
		self.transformations.TransformUnison(inputs, outputs, self.CallerClass.Channels, self.CallerClass.ChannelFirst,
											self.CallerClass.InputSizeX, self.CallerClass.InputSizeY,
											self.CallerClass.OutputSizeX, self.CallerClass.OutputSizeY)

		if 0 < self.CallerClass.NoiseRange:
			for input, normalizer in zip(inputs, self.InputNormalizers):
				self.transformations.Noise(input, self.CallerClass.NoiseRange, self.CallerClass.NoiseType,
											self.CallerClass.NoisePercentage, normalizer)
		
		if len(inputs) == 1:
			X = inputs[0].copy()
		else:
			X = np.stack(inputs, axis=0)
		
		if len(outputs) == 1:
			Y = outputs[0].copy()
		else:
			Y = np.stack(outputs, axis=0)
		
		return {'input': X, 'output': Y}





	def __len__(self):
		return self.Length

	def getNbBatchPerEpoch(self):
		return self.nbBatchPerEpoch











# ------------------------------------------------------------ PyTorch Multiple Datasets ------------------------------------------------------------
class PyTorchMultipleDatasets(Dataset):

	def __init__(self, Datasets: list, BatchSize: int):
	
		super().__init__()
		
		if len(Datasets) <= 1: raise Exception("Number of Datasets <= 1.")
		self.Datasets = Datasets
		
		self.FakeStarts = np.ndarray(shape=(len(Datasets)), dtype=np.uint32)
		self.Randomize = np.ndarray(shape=(len(Datasets)), dtype=bool)
		self.RealLength = 0
		self.FakeLength = 0
		index = pos = 0
		for dataset in Datasets:
			if not isinstance(dataset["Dataset"], Dataset):
				raise Exception("The Dataset object in the element " + str(index) + " of the list must be an instance of Dataset.")
			
			#if dataset["Dataset"].CallerClass.nbCropPerImage != 1:
			#	raise Exception("The Dataset " + str(index) + " is set up with more than 1 crop per image.")
			
			self.RealLength += dataset["Dataset"].__len__()
			self.FakeLength += dataset["Length"]
			
			self.FakeStarts[index] = pos
			pos += dataset["Length"]
			
			if dataset["Dataset"].__len__() != dataset["Length"]: self.Randomize[index] = True
			else: self.Randomize[index] = False
			
			index += 1
		
		if self.FakeLength < BatchSize:
			for dataset in Datasets:
				print("Warning - PyTorchMultipleDatasets - Real = %d, Fake = %d" % (dataset["Dataset"].__len__(), dataset["Length"]))
			raise Exception("The batch size is greater than the number of data: %d < %d" % (self.FakeLength, BatchSize))
		
		self.RealIndex = np.ndarray(shape=(self.RealLength), dtype=np.uint32)
		self.FakeIndex = np.ndarray(shape=(self.FakeLength), dtype=np.uint32)
		
		self.RealLength = 0
		self.FakeLength = 0
		index = 0
		for dataset in Datasets:
			if not isinstance(dataset["Dataset"], Dataset):
				raise Exception("The Dataset object in the element " + str(index) + " of the list must be an instance of Dataset.")
			RealLen = dataset["Dataset"].__len__()
			FakeLen = dataset["Length"]
			
			self.RealIndex[self.RealLength:self.RealLength+RealLen] = index
			self.FakeIndex[self.FakeLength:self.FakeLength+FakeLen] = index
			
			self.RealLength += RealLen
			self.FakeLength += FakeLen
			index += 1
		
		self.nbBatchPerEpoch = int(self.FakeLength/BatchSize)
		
		self.rand = random.Random()


	def __getitem__(self, index):
		
		datasetID = self.FakeIndex[index]
		dataset = self.Datasets[datasetID]["Dataset"]
		
		if self.Randomize[datasetID] == True: Index = self.rand.randint(0, dataset.__len__()-1)
		else: Index = index - self.FakeStarts[datasetID]
		
		return dataset.__getitem__(Index)
	

	def __len__(self):
		return self.FakeLength


	def getNbBatchPerEpoch(self):
		return self.nbBatchPerEpoch

