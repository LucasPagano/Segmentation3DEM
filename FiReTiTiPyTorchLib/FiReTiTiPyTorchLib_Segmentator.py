import math
import numpy
import os
import pandas
import random
import sys
import threading
import time
import torch

from horology import timed, Timing

from PIL import Image, ImageDraw

from torchvision.transforms import functional as F

import ImagesIO
import ImageTools
import Tools






class _SingleIterator(object):
	
	def __init__(self, Image, CropSize: int, BorderEffectSize: int, Iterator: bool=False, verbose=False):
		
		self.Image = Image
		self.Width, self.Height, self.Channels, self.First = ImageTools.Dimensions(Image)
		
		if verbose:
			print("Dimensions = " + str(self.Width) + " x " + str(self.Height) + " x " + str(self.Channels) + ", channel first=" + str(self.First))
		
		self.CropSize = CropSize
		self.Shift = CropSize - 2*BorderEffectSize
		
		self.Iterator = Iterator
		
		self.Reset()
		
		self.lock = threading.Lock()
		self.PositionsGenerator = self.FlowIndex()
	
	
	def Reset(self):
		self.PosX = self.PosY = 0
	
	
	def FlowIndex(self):
		self.Reset()
		
		Xwait = False
		Ywait = False
		
		while 1:
			if not Xwait and self.Width <= self.PosX+self.CropSize:
				self.PosX = 0
				wait = False
				
				if Ywait == False:
					self.PosY += self.Shift
				else:
					return #-1, -1 # raise StopIteration
				
				if self.Height < self.PosY+self.CropSize:
					self.PosY = self.Height - self.CropSize
					Ywait = True
			

			yield self.PosX, self.PosY
			
			
			if Xwait == False:
				self.PosX += self.Shift
			
			if self.Width == self.PosX+self.CropSize:
				if Xwait == True: Xwait = False
				else: Xwait = True
			
			if self.Width < self.PosX+self.CropSize:
				self.PosX = self.Width - self.CropSize
				Xwait = True


	def __iter__(self):
		return self # needed if we want to do something like: for x, y in data_gen.flow(...):
	
	
	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)





class Segment(object):
	""" This class evaluates results.
		Args:
		
		Examples:
			...
		
		
		TODO:
			...
		"""
	
	
	
	def __init__(self, verbose: bool=True):
		""" Initialization.
			Args:
				verbose (bool, optional): Defaults to True.
		"""
		self.__Reset__()
		
		self.verbose = verbose
		
		self.rand = random.Random()
	
	
	def __Reset__(self):
		"..."
		return




	@timed
	def PyTorch(self, ImagesDirPath: str, Model, Device, CropSize: int, BorderEffectSize: int,
				InputNormalizer, BatchSize: int, OutputsNormalizers: list, ResultsDirPath: str):
		""" This function segment all the images in the given directory using the given model.
			Args:
				ImagesDirPath  (str): The directory containing the images to segment.
				Model (object): The model to use.
				Device (object): The type of device to use. Usually, the answer of 'torch.device("cuda" if torch.cuda.is_available() else "cpu")'
				CropSize (int): The crop dimension. Windows/crops of dimensions (CropSize,CropSize) will be generated and segmented.
				BorderEffectSize (int): The size of the border effect.
				InputNormalizer (object): The normalizer to use to normalize the input image.
				OutputsNormalizers (list): The normalizers to use to denormalize the output/result tensors before image conversion.
				ResultsDirPath (str): The directory to save the segmented images.
		"""
		
		if CropSize <= 0:
			raise Exception("CropSize <= 0... you Moron!")
		
		if CropSize <= 2*BorderEffectSize:
			raise Exception("CropSize <= 2 x BorderEffectSize... you Moron!")
		
		if OutputsNormalizers is None or len(OutputsNormalizers) == 0:
			nbOuputs = 1
		else:
			nbOuputs = len(OutputsNormalizers)
			
		bes = BorderEffectSize
		csmbes = CropSize - BorderEffectSize
		
		if nbOuputs == 1:
			os.makedirs(ResultsDirPath, exist_ok=True)
		else:
			for o in range(nbOuputs):
				os.makedirs(ResultsDirPath + "/Out " + str(o), exist_ok=True)
		
		imnames = ImagesIO.FindImages(ImagesDirPath, NamesOnly=True, verbose=False)
		images = ImagesIO.LoadImages(ImagesDirPath, ChannelFirst=True, ReturnImagesList=False, verbose=False)
		nbImages = len(images)
		
		if self.verbose:
			print(" - %d images in %s\n\n" % (nbImages, ImagesDirPath))
			sys.stdout.flush()
		
		
		for i in range(nbImages):
			if self.verbose:
				print(" - Segmenting image: '" + str(imnames[i]) + str("'"))
			
			image = images[i]
			width, height, channels, first = ImageTools.Dimensions(image)
			
			result = None #numpy.ndarray(shape=(nbOuputs, channels, height, width), dtype=numpy.float32)
			
			batch = numpy.ndarray(shape=(BatchSize, channels, CropSize, CropSize), dtype=numpy.float32)
			
			if InputNormalizer is not None:
				InputNormalizer.Normalize(image)

			iter = _SingleIterator(image, CropSize, BorderEffectSize, True, verbose=self.verbose) # Iterates throughout the image.
			coordinates = []
			try:
				while True : # Get all the coordinates.
					X, Y = next(iter.PositionsGenerator)
					#print(str(X) + " " + str(Y))
					coordinates.append((X, Y))
			except StopIteration:
				pass
			
			length = len(coordinates)
			pos = 0
			while pos < length:
				start = pos
				pos += BatchSize
				end = min(pos, length)
				
				for c, index in zip(range(start, end), range(end-start)):
					X, Y = coordinates[c]
					batch[index] = image[:, Y:Y+CropSize, X:X+CropSize]
				
				torchbatch = torch.tensor(batch, device=Device).float()
				
				predictions = Model(torchbatch) # Model at work!!!
				#predictions = torch.tensor([8, 1, 512, 512], dtype=torch.float32)
				
				if result is None:
					if isinstance(predictions, list):
						result = numpy.ndarray(shape=(nbOuputs, predictions[0].shape[len(predictions[0].shape)-3], height, width), dtype=numpy.float32)
					else:
						result = numpy.ndarray(shape=(nbOuputs, predictions.shape[len(predictions.shape)-3], height, width), dtype=numpy.float32)
					#print("result = " + str(result.shape))
				
				#print("predictions = " + str(predictions.size()))
				#print("predictionsLength = " + str(len(predictions)) + ", shape = " + str(predictions[0].size()))
				#print("torchbatch = " + str(torchbatch.size()))
				
				for c, index in zip(range(start, end), range(end-start)):
					X, Y = coordinates[c]
					
					if isinstance(predictions, list):
						out = 0
						for pred in predictions:
							if X == 0 or Y == 0 or X+CropSize == width or Y+CropSize == height:
								result[out, :, Y:Y+CropSize, X:X+CropSize] = pred[index].detach().cpu().numpy()
							else:
								result[out, :, Y+bes:Y+csmbes, X+bes:X+csmbes] = pred[index, :, bes:csmbes, bes:csmbes].detach().cpu().numpy()
							out += 1
					else:
						if nbOuputs == 1:
							if X == 0 or Y == 0 or X+CropSize == width or Y+CropSize == height:
								result[0, :, Y:Y+CropSize, X:X+CropSize] = predictions[index].detach().cpu().numpy()
							else:
								result[0, :, Y+bes:Y+csmbes, X+bes:X+csmbes] = predictions[index, :, bes:csmbes, bes:csmbes].detach().cpu().numpy()
						elif 1 < nbOuputs:
							if X == 0 or Y == 0 or X+CropSize == width or Y+CropSize == height:
								result[:, :, Y:Y+CropSize, X:X+CropSize] = predictions[index].detach().cpu().numpy()
							else:
								result[:, :, Y+bes:Y+csmbes, X+bes:X+csmbes] = predictions[index, :, :, bes:csmbes, bes:csmbes].detach().cpu().numpy()
						else:
							raise Exception("nbOuputs = " + str(nbOuputs) + ". Must not occur... you Moron!")

			if OutputsNormalizers is not None:
				for normalizer, o in zip(OutputsNormalizers, range(len(OutputsNormalizers))):
					normalizer.Denormalize(result[o])

			if nbOuputs == 1:
				ImagesIO.Write(result[0], True, ResultsDirPath + "/" + imnames[i].replace('.tiff','.png').replace('.tif','.png'))
			else:
				for o in range(nbOuputs):
					ImagesIO.Write(result[o], True, ResultsDirPath + "/Out " + str(o)+"/"+imnames[i].replace('.tiff','.png').replace('.tif','.png'))
			
			"""
			with Timing(name="Time = "):
				print("Equal = " + str((image == result).all()))
			with Timing(name="Time = "):
				print("Equal = " + str(numpy.array_equal(image, result)))
			"""







	@timed
	def MaskRCNN(self, ImagesDirPath: str, ImageNameFilter: str, Model, Device, CropSize: int, BorderEffectSize: int,
				InputsNormalizers: list, BatchSize: int, CheckFullOverlap: int, ResultsDirPath: str, SaveIndividualObject: bool,
				BoxProbability: float=0.13, MaskThreshold: float=0.5,
				Margin: int=1):
		""" This function segment all the images in the given directory using the given model.
			Args:
				ImagesDirPath (str): The directory containing the images to segment.
				ImageNameFilter (str): A way to keep only a given type of images. If not none, then the images name must contain this string to be processed.
				Model (object): The model to use.
				Device (object): The type of device to use. Usually, the answer of 'torch.device("cuda" if torch.cuda.is_available() else "cpu")'
				CropSize (int): The crop dimension. Windows/crops of dimensions (CropSize,CropSize) will be generated and segmented.
				BorderEffectSize (int): The size of the border effect.
				InputNormalizer (object): The normalizer to use to normalize the input image.
				CheckFullOverlap (int): Check for full overlaps and remove them? This value is the error accepted to consider overlap.
				ResultsDirPath (str): The directory to save the segmented images.
				SaveIndividualObject (bool): Save each object into a separate image? If True, a directory "Objetcs" will be created.
				BoxProbability (float): The threshold (minimum probability) to consider a box as valid candidated.
				MaskThreshold (float): The threshold to use on each mask.
				Margin (int): The margin to add around each box.
		"""
		
		print("\nStarting segmentation with MaskRCNN.")
		
		if CropSize <= 0:
			raise Exception("CropSize <= 0... you Moron!")
		
		if CropSize <= 2*BorderEffectSize:
			raise Exception("CropSize <= 2 x BorderEffectSize... you Moron!")
		
		if BoxProbability < 0.0 or 1.0 < BoxProbability:
			raise Exception("BoxProbability < 0.0 or 1.0 < BoxProbability... you Moron!")
		
		if MaskThreshold < 0.0 or 1.0 < MaskThreshold:
			raise Exception("MaskThreshold < 0.0 or 1.0 < MaskThreshold... you Moron!")
		
		if InputsNormalizers is not None and len(InputsNormalizers) != 1:
			raise Exception("If not None, a single element is expected in InputsNormalizers.")
		
		bes = BorderEffectSize
		csmbes = CropSize - BorderEffectSize
		
		os.makedirs(ResultsDirPath, exist_ok=True)
		
		imnames = ImagesIO.FindImages(ImagesDirPath, NamesOnly=True, verbose=False)
		impaths = ImagesIO.FindImages(ImagesDirPath, NamesOnly=False, verbose=False)
		
		if ImageNameFilter is not None:
			todelete = []
			i = 0
			for name in imnames:
				if ImageNameFilter not in name:
					todelete.append(i)
				i += 1
			todelete.reverse()
			for num in todelete:
				del imnames[num]
				del impaths[num]
			
		
		nbImages = len(imnames)
		for i in range(nbImages):
			if self.verbose:
				print(" - Segmenting image: '" + str(imnames[i]) + str("'"))
			
			imname = imnames[i].replace('.tiff', '').replace('.tif', '').replace('.png', '')
			
			if SaveIndividualObject == True:
				objdir = ResultsDirPath + "/" + imname + " - Objects/"
				os.makedirs(objdir, exist_ok=True)
			
			image = ImagesIO.LoadImagesList([impaths[i]], ChannelFirst=True, ReturnImagesList=False, verbose=self.verbose)[0]
			width, height, channels, first = ImageTools.Dimensions(image)
			
			imclone = numpy.copy(image)
			
			if channels == 1:
				image = numpy.tile(image, (3, 1, 1))
			width, height, channels, first = ImageTools.Dimensions(image)
			wmc = width  - CropSize
			hmc = height - CropSize
			
			imbox = image.astype(numpy.uint16, copy=True)
			imbox = imbox / 256
			imbox = imbox.astype(numpy.uint8, copy=True)
			imbox = numpy.moveaxis(imbox, 0, -1)
			
			if InputsNormalizers is not None:
				InputsNormalizers[0].Normalize(image)
			
			resBoxes  = Image.fromarray(imbox, mode='RGB')
			drawBoxes = ImageDraw.Draw(resBoxes) # To draw the boxes.
			
			resCandidates  = Image.fromarray(imbox, mode='RGB')
			drawCandidates = ImageDraw.Draw(resCandidates) # To draw the boxes.
			
			resProbabilities  = Image.fromarray(imbox, mode='RGB')
			drawProbabilities = ImageDraw.Draw(resProbabilities) # To draw the boxes.
			
			resLabels = numpy.ndarray(shape=(height, width), dtype=numpy.int32)
			resLabels.fill(0)
			
			resForeground = numpy.ndarray(shape=(height, width), dtype=numpy.uint8)
			resForeground.fill(0)
			
			batch = numpy.ndarray(shape=(BatchSize, 3, CropSize, CropSize), dtype=numpy.float32)

			iter = _SingleIterator(image, CropSize, BorderEffectSize, True, verbose=False) # Iterates throughout the image.
			coordinates = []
			
			while True : # Get all the coordinates.
				try:
					X, Y = next(iter.PositionsGenerator)
				except StopIteration:
					break
				coordinates.append((X, Y))
			
			MemoryBoxes = []
			num = 1
			length = len(coordinates)
			pos = 0
			while pos < length:
				start = pos
				pos += BatchSize
				end = min(pos, length)
				
				for c, index in zip(range(start, end), range(end-start)):
					X, Y = coordinates[c]
					batch[index] = image[:, Y:Y+CropSize, X:X+CropSize]
				
				torchbatch = torch.tensor(batch, device=Device).float()
				
				predictions = Model(torchbatch) # Model at work!!!
				
				for c, index in zip(range(start, end), range(end-start)):
					X, Y = coordinates[c]
					
					Proceed = True
					try:
						boxes  = predictions[index]["boxes"]
					except IndexError:
						Proceed = False
					
					if Proceed == True:
						scores = predictions[index]["scores"]
						#labels = predictions[index]["labels"]
						masks  = predictions[index]["masks"]
						
						for bms in range(boxes.shape[0]-1, -1, -1): #for box, mask, score in reversed(zip(boxes, masks, scores)):
							box = boxes[bms].data.cpu().numpy()
							mask = masks[bms].data.cpu().numpy()
							proba = scores[bms].data.cpu().numpy()
							
							xmin, ymin = int(box[0]), int(box[1])
							xmax, ymax = int(box[2]), int(box[3])
							shape = [(X+xmin, Y+ymin), (X+xmax, Y+ymax)]
							boxshape = [X+xmin, Y+ymin, X+xmax, Y+ymax]
							
							color = self.RandomColor()
							drawCandidates.rectangle(shape, outline=color)
							drawCandidates.text(shape[0], str(proba)[1:4], fill=color)
							
							if xmax <= xmin: # TODO remove after intensive testing.
								Exception("X issue => " + str(xmin) + " vs " + str(xmax))
							if ymax <= ymin:
								Exception("Y issue => " + str(ymin) + " vs " + str(ymax))
							
							if 0 <= CheckFullOverlap: # Goes up because main loop (bms) goes down.
								bmsup = bms + 1
								while BoxProbability <= proba and bmsup < boxes.shape[0]:
									if self.areBoxesMatching(box, boxes[bmsup].data.cpu().numpy(), CheckFullOverlap) == True:
										proba = 0
									else:
										bmsup += 1
							
							
							if BoxProbability <= proba: # Filtering out the one touching the borders.
								go = False
								if X == 0:
									if csmbes < xmax: # Everything too far on the right is out.
										go = False
									elif csmbes < ymax:
										if Y+CropSize == height: # Top Left corner
											go = True
										else:
											go = False
									elif ymin == 0: # Touch the border of the tile, priority to the previous tile.
										if Y == 0: # Bottom Left corner
											go = True
										else:
											go = False
									else:
										go = True
								elif Y == 0:
									if csmbes < ymax: # Everything too far on the top is out.
										go = False
									elif csmbes < xmax:
										if X+CropSize == width: # Bottom right corner
											go = True
										else:
											go = False
									elif xmin == 0: # Touch the border of the tile, priority to the previous tile.
										if X == 0: # Bottom Left corner. Useless, already processed by X == 0.
											go = True
										else:
											go = False
									else:
										go = True
								elif X+CropSize == width:
									if csmbes < ymax:
										if Y+CropSize == height: # Top Left corner
											go = True
										else:
											go = False # Priority to next crop.
									elif xmin == 0 or ymin == 0: # Touch the border of the tile, priority to the previous tile.
										go = False
									#elif csmbes <= xmax: go = True # In the uncertain zone, but it's the last crop.
									else:
										go = True
								elif Y+CropSize == height:
									if csmbes < xmax:
										go = False # Priority to next crop. Other cases already processed.
									elif xmin == 0 or ymin == 0: # Touch the border of the tile, priority to the previous tile.
										go = False
									else:
										go = True
								else:
									if csmbes < xmax or csmbes < ymax: # right and upper are discarded. Priroty to the next tiles.
										go = False
									elif xmin == 0 or ymin == 0: # Touch the border of the tile, priority to the previous tile.
										go = False
									else:
										go = True
								
								
								if go == True and self.isContainXXX({"Box" : boxshape, "Proba" : proba}, MemoryBoxes, CheckFullOverlap) == False:
									miny = max(0, boxshape[1]-Margin)
									maxy = min(height, boxshape[3]+1+Margin)
									minx = max(0, boxshape[0]-Margin)
									maxx = min(width, boxshape[2]+1+Margin)
									
									miny2 = max(0, ymin-Margin)
									maxy2 = min(CropSize, ymax+1+Margin)
									minx2 = max(0, xmin-Margin)
									maxx2 = min(CropSize, xmax+1+Margin)
									
									if maxx-minx != maxx2-minx2:
										if maxx-minx < maxx2-minx2: maxx2 -= (maxx2-minx2) - (maxx-minx)
										else: maxx -= (maxx-minx) - (maxx2-minx2)
									
									if maxy-miny != maxy2-miny2:
										if maxy-miny < maxy2-miny2: maxy2 -= (maxy2-miny2) - (maxy-miny)
										else: maxy -= (maxy-miny) - (maxy2-miny2)
										
									crop = numpy.copy(imclone[:, miny:maxy, minx:maxx])
									
									rawmask = numpy.copy(mask[:, miny2:maxy2, minx2:maxx2])
									
									MemoryBoxes.append({"Box" : [minx, miny, maxx, maxy], "Crop" : crop, "Proba" : proba, "Mask" : rawmask})
									num += 1
			
			
			FinalBoxes = sorted(MemoryBoxes, key=lambda i : i["Proba"], reverse=True)
			
			num = 1
			for box in FinalBoxes:
				bb = box["Box"]
				proba = box["Proba"]
				shape = [(bb[0], bb[1]), (bb[2], bb[3])]
				mask = box["Mask"]
				crop = box["Crop"]
				
				resForeground[bb[1]:bb[3], bb[0]:bb[2]] = numpy.maximum(resForeground[bb[1]:bb[3], bb[0]:bb[2]], mask[0,:,:]*255.0)
				
				W, H = mask.shape[2], mask.shape[1]
				for y in range(H):
					for x in range(W):
						if MaskThreshold <= mask[:, y, x]:
							if resLabels[bb[1]+y, bb[0]+x] == 0:
								resLabels[bb[1]+y, bb[0]+x] = num
						else:
							crop[:, y, x] = 0
				
				color = self.RandomColor()
				
				drawBoxes.rectangle(shape, outline=color)
				drawBoxes.text(shape[0], str(num), fill=color)
				
				drawProbabilities.rectangle(shape, outline=color)
				drawProbabilities.text(shape[0], str(proba)[1:4], fill=color)
				
				if SaveIndividualObject == True:
					ImagesIO.Write(crop, True, objdir + "/" + str(num) + "_" + str(bb[0]) + "x" + str(bb[1]) + ".png")
				
				num += 1
			
			
			MemoryBoxes.clear()
			FinalBoxes.clear()
			
			ext = " - Labels.png"
			ImagesIO.Write(resLabels, True, ResultsDirPath + "/" + imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))
			w, h = resBoxes.size
			ext = " - Boxes.png"
			resBoxes.save(ResultsDirPath + "/" + imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))
			ext = " - Candidates.png"
			resCandidates.save(ResultsDirPath + "/" + imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))
			ext = " - Probabilities.png"
			resProbabilities.save(ResultsDirPath + "/" + imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))
			ext = " - Foreground.png"
			ImagesIO.Write(resForeground.astype(numpy.uint8), True, ResultsDirPath + "/" + imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))









	@timed
	def MaskRCNN3(self, ImagesDirPath: str, Model, Device, CropSize: int, BorderEffectSize: int, InputsNormalizers: list, BatchSize: int,
						CheckFullOverlap: int, ResultsDirPath: str, SaveIndividualObject: bool, Threshold: float=0.5, Margin: int=1):
		""" This function segment all the images in the given directory using the given model.
			Args:
				ImagesDirPath (str): The directory containing the images to segment.
				Model (object): The model to use.
				Device (object): The type of device to use. Usually, the answer of 'torch.device("cuda" if torch.cuda.is_available() else "cpu")'
				CropSize (int): The crop dimension. Windows/crops of dimensions (CropSize,CropSize) will be generated and segmented.
				BorderEffectSize (int): The size of the border effect.
				InputNormalizer (list): The normalizer to use to normalize the input image.
				CheckFullOverlap (int): Check for full overlaps and remove them? This value is the error accepted to consider overlap.
				ResultsDirPath (str): The directory to save the segmented images.
				SaveIndividualObject (bool): Save each object into a separate image? If True, a directory "Objetcs" will be created.
				Threshold (int): The threshold to use on each box/mask probability.
				Margin (int): The margin to add around each box.
		"""
		raise Exception("Fix BoxProbability / MaskThreshold bug like in MaskRCNN1")
		print("\nStarting segmentation with MaskRCNN with multiple inputs.")
		
		if CropSize <= 0:
			raise Exception("CropSize <= 0... you Moron!")
		
		if CropSize <= 2*BorderEffectSize:
			raise Exception("CropSize <= 2 x BorderEffectSize... you Moron!")
		
		if Threshold < 0.0 or 1.0 < Threshold:
			raise Exception("Threshold < 0.0 or 1.0 < Threshold... you Moron!")
		
		if InputsNormalizers is not None and len(InputsNormalizers) != 3:
			raise Exception("If not None, 3 element are expected in InputsNormalizers.")
			
		bes = BorderEffectSize
		csmbes = CropSize - BorderEffectSize
		
		os.makedirs(ResultsDirPath, exist_ok=True)
		
		dirs = [dir for dir in os.listdir(ImagesDirPath) if os.path.isdir(os.path.join(ImagesDirPath, dir))]
		dirs.sort()
		if len(dirs) != 3:
			raise Exception(str(len(dirs)) + " found, 3 expected.")
		else:
			print("3 directories / inputs found in %s" % (ImagesDirPath))
		
		imnames = []
		impaths = []
		for dir in dirs:
			path = os.path.join(ImagesDirPath, dir)
			names = ImagesIO.FindImages(path, NamesOnly=True, verbose=False)
			imnames.append(names)
			impaths.append(ImagesIO.FindImages(path, NamesOnly=False, verbose=False))
			print(" - %d image(s) found in %s" % (len(names), path))
		
		nb = len(imnames[0])
		for names in imnames:
			if nb != len(names):
				raise Exception("The input directories have a different number of images.")
		
		
		nbImages = len(imnames[0])
		for i in range(nbImages):
			if self.verbose:
				print(" - Segmenting image: '" + str(imnames[0][i]) + "'")
			
			imname = imnames[0][i].replace('.tiff', '').replace('.tif', '').replace('.png', '')
			
			if SaveIndividualObject == True:
				objdir = ResultsDirPath + "/" + imname + " - Objects/"
				os.makedirs(objdir, exist_ok=True)
			
			inputs = []
			for input in impaths:
				inputs.append(ImagesIO.LoadImagesList([input[i]], ChannelFirst=True, ReturnImagesList=False, verbose=self.verbose)[0][0])
			image = numpy.stack(inputs, axis=0)
			
			imclone = numpy.copy(image)
			
			width, height, channels, first = ImageTools.Dimensions(image)
			wmc = width  - CropSize
			hmc = height - CropSize
			
			imbox = image.astype(numpy.uint16, copy=True)
			imbox = imbox / 256
			imbox = imbox.astype(numpy.uint8, copy=False)
			imbox = numpy.moveaxis(imbox, 0, -1)
			
			if InputsNormalizers is not None:
				for num, normalizer in enumerate(InputsNormalizers):
					normalizer.Normalize(image[num])
			
			resBoxes  = Image.fromarray(imbox, mode='RGB')
			drawBoxes = ImageDraw.Draw(resBoxes) # To draw the boxes.
			
			resCandidates  = Image.fromarray(imbox, mode='RGB')
			drawCandidates = ImageDraw.Draw(resCandidates) # To draw the boxes.
			
			resProbabilities  = Image.fromarray(imbox, mode='RGB')
			drawProbabilities = ImageDraw.Draw(resProbabilities) # To draw the boxes.
			
			resLabels = numpy.ndarray(shape=(height, width), dtype=numpy.int32)
			resLabels.fill(0)
			
			resForeground = numpy.ndarray(shape=(height, width), dtype=numpy.uint8)
			resForeground.fill(0)
			
			batch = numpy.ndarray(shape=(BatchSize, 3, CropSize, CropSize), dtype=numpy.float32)

			iter = _SingleIterator(image, CropSize, BorderEffectSize, True, verbose=False) # Iterates throughout the image.
			coordinates = []
			while True : # Get all the coordinates.
				try:
					X, Y = next(iter.PositionsGenerator)
				except StopIteration:
					break
				coordinates.append((X, Y))
			
			
			MemoryBoxes = []
			num = 1
			length = len(coordinates)
			pos = 0
			while pos < length:
				start = pos
				pos += BatchSize
				end = min(pos, length)
				
				for c, index in zip(range(start, end), range(end-start)):
					X, Y = coordinates[c]
					batch[index] = image[:, Y:Y+CropSize, X:X+CropSize]
				
				torchbatch = torch.tensor(batch, device=Device).float()
				
				predictions = Model(torchbatch) # Model at work!!!
				
				for c, index in zip(range(start, end), range(end-start)):
					X, Y = coordinates[c]
					
					boxes  = predictions[index]["boxes"]
					scores = predictions[index]["scores"]
					#labels = predictions[index]["labels"]
					masks  = predictions[index]["masks"]
					
					for bms in range(boxes.shape[0]-1, -1, -1): #for box, mask, score in reversed(zip(boxes, masks, scores)):
						box = boxes[bms].data.cpu().numpy()
						mask = masks[bms].data.cpu().numpy()
						proba = scores[bms].data.cpu().numpy()
						
						xmin, ymin = int(box[0]), int(box[1])
						xmax, ymax = int(box[2]), int(box[3])
						shape = [(X+xmin, Y+ymin), (X+xmax, Y+ymax)]
						boxshape = [X+xmin, Y+ymin, X+xmax, Y+ymax]
						
						color = self.RandomColor()
						drawCandidates.rectangle(shape, outline=color)
						drawCandidates.text(shape[0], str(proba)[1:4], fill=color)
						
						if xmax <= xmin: # TODO remove after intensive testing.
							Exception("X issue => " + str(xmin) + " vs " + str(xmax))
						if ymax <= ymin:
							Exception("Y issue => " + str(ymin) + " vs " + str(ymax))
						
						if 0 <= CheckFullOverlap: # Goes up because main loop (bms) goes down.
							bmsup = bms + 1
							while Threshold <= proba and bmsup < boxes.shape[0]:
								if self.areBoxesMatching(box, boxes[bmsup].data.cpu().numpy(), CheckFullOverlap) == True:
									proba = 0
								else:
									bmsup += 1
						
						
						if Threshold <= proba: # Filtering out the one touching the borders.
							go = False
							if X == 0:
								if csmbes < xmax: # Everything too far on the right is out.
									go = False
								elif csmbes < ymax:
									if Y+CropSize == height: # Top Left corner
										go = True
									else:
										go = False
								elif ymin == 0: # Touch the border of the tile, priority to the previous tile.
									if Y == 0: # Bottom Left corner
										go = True
									else:
										go = False
								else:
									go = True
							elif Y == 0:
								if csmbes < ymax: # Everything too far on the top is out.
									go = False
								elif csmbes < xmax:
									if X+CropSize == width: # Bottom right corner
										go = True
									else:
										go = False
								elif xmin == 0: # Touch the border of the tile, priority to the previous tile.
									if X == 0: # Bottom Left corner. Useless, already processed by X == 0.
										go = True
									else:
										go = False
								else:
									go = True
							elif X+CropSize == width:
								if csmbes < ymax:
									if Y+CropSize == height: # Top Left corner
										go = True
									else:
										go = False # Priority to next crop.
								elif xmin == 0 or ymin == 0: # Touch the border of the tile, priority to the previous tile.
									go = False
								#elif csmbes <= xmax: go = True # In the uncertain zone, but it's the last crop.
								else:
									go = True
							elif Y+CropSize == height:
								if csmbes < xmax:
									go = False # Priority to next crop. Other cases already processed.
								elif xmin == 0 or ymin == 0: # Touch the border of the tile, priority to the previous tile.
									go = False
								else:
									go = True
							else:
								if csmbes < xmax or csmbes < ymax: # right and upper are discarded. Priroty to the next tiles.
									go = False
								elif xmin == 0 or ymin == 0: # Touch the border of the tile, priority to the previous tile.
									go = False
								else:
									go = True
							
							
							if go == True and self.isContainXXX({"Box" : boxshape, "Proba" : proba}, MemoryBoxes, CheckFullOverlap) == False:
								miny = max(0, boxshape[1]-Margin)
								maxy = min(height, boxshape[3]+1+Margin)
								minx = max(0, boxshape[0]-Margin)
								maxx = min(width, boxshape[2]+1+Margin)
								
								miny2 = max(0, ymin-Margin)
								maxy2 = min(CropSize, ymax+1+Margin)
								minx2 = max(0, xmin-Margin)
								maxx2 = min(CropSize, xmax+1+Margin)
								
								if maxx-minx != maxx2-minx2:
									if maxx-minx < maxx2-minx2: maxx2 -= (maxx2-minx2) - (maxx-minx)
									else: maxx -= (maxx-minx) - (maxx2-minx2)
								
								if maxy-miny != maxy2-miny2:
									if maxy-miny < maxy2-miny2: maxy2 -= (maxy2-miny2) - (maxy-miny)
									else: maxy -= (maxy-miny) - (maxy2-miny2)
									
								crop = numpy.copy(imclone[:, miny:maxy, minx:maxx])
								
								rawmask = numpy.copy(mask[:, miny2:maxy2, minx2:maxx2])
								
								MemoryBoxes.append({"Box" : [minx, miny, maxx, maxy], "Crop" : crop, "Proba" : proba, "Mask" : rawmask})
								num += 1
			
			
			FinalBoxes = sorted(MemoryBoxes, key=lambda i : i["Proba"], reverse=True)
			
			num = 1
			for box in FinalBoxes:
				bb = box["Box"]
				proba = box["Proba"]
				shape = [(bb[0], bb[1]), (bb[2], bb[3])]
				mask = box["Mask"]
				crop = box["Crop"]
				
				resForeground[bb[1]:bb[3], bb[0]:bb[2]] = numpy.maximum(resForeground[bb[1]:bb[3], bb[0]:bb[2]], mask[0,:,:]*255.0)
				
				W, H = mask.shape[2], mask.shape[1]
				for y in range(H):
					for x in range(W):
						if Threshold <= mask[:, y, x]:
							if resLabels[bb[1]+y, bb[0]+x] == 0:
								resLabels[bb[1]+y, bb[0]+x] = num
						else:
							crop[:, y, x] = 0
				
				color = self.RandomColor()
				
				drawBoxes.rectangle(shape, outline=color)
				drawBoxes.text(shape[0], str(num), fill=color)
				
				drawProbabilities.rectangle(shape, outline=color)
				drawProbabilities.text(shape[0], str(proba)[1:4], fill=color)
				
				if SaveIndividualObject == True:
					ImagesIO.Write(crop, True, objdir + "/" + str(num) + "_" + str(bb[0]) + "x" + str(bb[1]) + ".png")
				
				num += 1
			
			
			MemoryBoxes.clear()
			FinalBoxes.clear()
			
			ext = " - Labels.png"
			ImagesIO.Write(resLabels, True, ResultsDirPath + "/" + imname + ext)#s[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))
			w, h = resBoxes.size
			ext = " - Boxes.png"
			resBoxes.save(ResultsDirPath + "/" + imname + ext)#imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))
			ext = " - Candidates.png"
			resCandidates.save(ResultsDirPath + "/" + imname + ext)#imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))
			ext = " - Probabilities.png"
			resProbabilities.save(ResultsDirPath + "/" + imname + ext)#imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))
			ext = " - Foreground.png"
			ImagesIO.Write(resForeground.astype(numpy.uint8), True, ResultsDirPath + "/" + imname + ext)#imnames[i].replace('.tiff', ext).replace('.tif', ext).replace('.png', ext))






	def RandomColor(self):
		value = self.rand.randint(0, 7)
		if   value == 0: return "green"
		elif value == 1: return "cyan"
		elif value == 2: return "yellow"
		elif value == 3: return "maroon"
		elif value == 4: return "purple"
		elif value == 5: return "red"
		return "blue"
	
	
	def FindMatchingBox(self, box, boxes, margin: int):
		for boxe in boxes:
			if self.areBoxesMatching(box, boxe, margin) == True: return True
		return False
	
	
	def areBoxesMatching(self, box1, box2, margin: int):
		for i in range(4):
			if margin < math.fabs(box1[i]-box2[i]): return False
		return True
	
	
	def isContainXXX(self, box, boxes, margin: int):
		b = box["Box"]
		p = box["Proba"]
		
		i = 0
		while i < len(boxes):
			if self.isContained(b, boxes[i]["Box"], margin) == True:
				if p < boxes[i]["Proba"]: return True
				else: del boxes[i]
			else: i += 1
		
		i = 0
		while i < len(boxes):
			if self.isContained(boxes[i]["Box"], b, margin) == True:
				if p < boxes[i]["Proba"]: return True
				else: del boxes[i]
			else: i += 1
		
		i = 0
		while i < len(boxes):
			if self.areBoxesMatching(boxes[i]["Box"], b, margin) == True:
				if p < boxes[i]["Proba"]: return True
				else: del boxes[i]
			else: i += 1
		
		return False
	
	
	def isContained(self, bx1, bx2, margin):
		if bx1[0] < bx2[0]-margin: return False
		if bx1[1] < bx2[1]-margin: return False
		if bx2[2]+margin < bx1[2]: return False
		if bx2[3]+margin < bx1[3]: return False
		return True
	
