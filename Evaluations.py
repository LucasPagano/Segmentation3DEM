import math
import numpy
import os
import pandas
import sys
import time

import ImagesIO
import ImageTools

from horology import timed, Timing




class Evaluations(object):
	
	def __init__(self, Threshold: int=128, MaxLevel: int=255, SaveImages: bool=True, verbose: bool=True):
		""" Initialization.
			Args:
				Threshold (int): Threshold value to consider a pixel as positive. Defaults to 128.
				MaxLevel (int): The maximum possible value in the images. Defaults to 255.
				SaveImages (bool, optional): Save the evaluation images? Defaults to True.
				verbose (bool, optional): Defaults to True.
		"""
		self.__Reset__()
		self.Threshold = Threshold
		self.MaxLevel = MaxLevel
		self.SaveImages = SaveImages
		self.verbose = verbose
	
	
	def __Reset__(self):
		pass


	
	"""
	@timed
	def BestSegmentation(self, PredictionsDirPath: str, GroundTruthsDirPath: str, SaveEvalutationsAs: str=None, SaveEvalutationImagesIn: str=None):
		 This function evaluates all the possible threshold to find the best segmentation results for all the images in the given directory.
			Args:
				PredictionsDirPath  (str): The directory containing the predicted images.
				GroundTruthsDirPath (str): The directory containing the ground truth images.
				SaveEvalutationsAs (str): The complete file path to save the evaluation values.
				SaveEvalutationImagesIn (str): If not None, the evaluation images will be saved in this directory.
		
		if SaveEvalutationImagesIn is not None:
			os.makedirs(SaveEvalutationImagesIn, exist_ok=True)
		
		predictions_names = ImagesIO.FindImages(PredictionsDirPath, True, False)
		groundtruths_names = ImagesIO.FindImages(GroundTruthsDirPath, True, False)
		
		if self.verbose:
			print(" - %d images in %s" % (len(predictions_names), PredictionsDirPath))
			print(" - %d images in %s" % (len(groundtruths_names), GroundTruthsDirPath))
			sys.stdout.flush()
		
		nbImages = len(predictions_names)
		if nbImages != len(groundtruths_names):
			raise Exception("Different number of images in the directories.")
		
		predictions = ImagesIO.LoadImages(PredictionsDirPath, ChannelFirst=True, ReturnImagesList=False, verbose=False)
		groundtruths = ImagesIO.LoadImages(GroundTruthsDirPath, ChannelFirst=True, ReturnImagesList=False, verbose=False)

		bestDice = 0.0
		bestThreshold = 0
		OldThreshold = self.Threshold
		
		for threshold in range(1, int(self.MaxLevel)):
			self.Threshold = threshold
			sumDice = 0.0
			nb = 0
			
			for i in range(nbImages):
				Accuracy, Sensitivity, Specificity, Precision, Dice = self.Segmentation(predictions[i], groundtruths[i], SaveImageAs=None)
				if not math.isnan(Dice):
					sumDice += Dice
					nb += 1

			sumDice /= float(nb)
			if bestDice < sumDice:
				bestThreshold = threshold
				bestDice = sumDice

		self.Threshold = bestThreshold
		if self.verbose:
			print("Best threshold for '" + PredictionsDirPath + "' equal " + str(bestThreshold) + ", with a dice coefficient of " + str(bestDice))
		self.Segmentations(PredictionsDirPath, GroundTruthsDirPath, SaveEvalutationsAs, SaveEvalutationImagesIn)

		self.Threshold = OldThreshold
	"""



	@timed
	def BestSegmentation(self, PredictionsDirPath: str, GroundTruthsDirPath: str, SaveEvalutationsAs: str=None, SaveEvalutationImagesIn: str=None):
		""" This function evaluates all the possible threshold to find the best segmentation results for all the images in the given directory.
			Args:
				PredictionsDirPath  (str): The directory containing the predicted images.
				GroundTruthsDirPath (str): The directory containing the ground truth images.
				SaveEvalutationsAs (str): The complete file path to save the evaluation values.
				SaveEvalutationImagesIn (str): If not None, the evaluation images will be saved in this directory.
		"""
		if SaveEvalutationImagesIn is not None:
			os.makedirs(SaveEvalutationImagesIn, exist_ok=True)
		
		predictions_names  = ImagesIO.FindImages(PredictionsDirPath, NamesOnly=True, verbose=False)
		groundtruths_names = ImagesIO.FindImages(GroundTruthsDirPath, NamesOnly=True, verbose=False)
		
		if self.verbose:
			print(" - %d images in %s" % (len(predictions_names), PredictionsDirPath))
			print(" - %d images in %s" % (len(groundtruths_names), GroundTruthsDirPath))
			sys.stdout.flush()
		
		nbImages = len(predictions_names)
		if nbImages != len(groundtruths_names):
			raise Exception("Different number of images in the directories.")
	
		predictions = ImagesIO.LoadImages(PredictionsDirPath, ChannelFirst=True, ReturnImagesList=False, verbose=False)
		groundtruths = ImagesIO.LoadImages(GroundTruthsDirPath, ChannelFirst=True, ReturnImagesList=False, verbose=False)

		step = int(math.sqrt((self.MaxLevel-1)/2.0))
		values = [val for val in range(0, self.MaxLevel, step)]
		if values[len(values)-1] != self.MaxLevel:
			values.append(self.MaxLevel)
		
		bestDice = 0.0
		bestThreshold = 0
		
		for threshold in values:
			sumDice = 0.0
			nb = 0
			
			for i in range(nbImages):
				Accuracy, Sensitivity, Specificity, Precision, Dice = Segmentation(predictions[i], groundtruths[i],
																					Threshold=threshold, SaveImageAs=None)
				if not math.isnan(Dice):
					sumDice += Dice
					nb += 1
			
			if 0 < nb:
				sumDice /= float(nb)
				if bestDice < sumDice:
					bestThreshold = threshold
					bestDice = sumDice
		
		
		avoid = bestThreshold
		rangemin = max(0, bestThreshold-step)
		rangemax = min(bestThreshold+step, self.MaxLevel)
		
		for threshold in list(range(rangemin,avoid)) + list(range(avoid+1,rangemax)): # Do not process value 'avoid-step', 'avoid', and 'avoid+setp'
			sumDice = 0.0
			nb = 0
			
			for i in range(nbImages):
				Accuracy, Sensitivity, Specificity, Precision, Dice = Segmentation(predictions[i], groundtruths[i],
																					Threshold=threshold, SaveImageAs=None)
				if not math.isnan(Dice):
					sumDice += Dice
					nb += 1
		
			if 0 < nb:
				sumDice /= float(nb)
				if bestDice < sumDice:
					bestThreshold = threshold
					bestDice = sumDice
		
		self.Threshold = bestThreshold
		if self.verbose:
			print("Best threshold for '" + PredictionsDirPath + "' equal " + str(bestThreshold) + ", with a dice coefficient of " + str(bestDice))
		self.Segmentations(PredictionsDirPath, GroundTruthsDirPath, bestThreshold, SaveEvalutationsAs, SaveEvalutationImagesIn)





	@timed
	def Segmentations(self, PredictionsDirPath: str, GroundTruthsDirPath: str, Threshold=None,
							SaveEvalutationsAs: str=None, SaveEvalutationImagesIn: str=None):
		""" This function evaluates the segmentation results for all the images in the given directory.
			Args:
				PredictionsDirPath  (str): The directory containing the predicted images.
				GroundTruthsDirPath (str): The directory containing the ground truth images.
				SaveEvalutationsAs (str): The complete file path to save the evaluation values.
				SaveEvalutationImagesIn (str): If not None, the evaluation images will be saved in this directory.
			Returns:
				...
		"""

		if SaveEvalutationImagesIn is not None:
			os.makedirs(SaveEvalutationImagesIn, exist_ok=True)
		
		if Threshold is not None: inThresh = Threshold
		else: inThresh = self.Threshold
		
		predictionsnames = ImagesIO.FindImages(PredictionsDirPath, NamesOnly=True, verbose=False)
		predictions = ImagesIO.FindImages(PredictionsDirPath, NamesOnly=False, verbose=False)
		
		groundtruths = ImagesIO.FindImages(GroundTruthsDirPath, NamesOnly=False, verbose=False)
		
		if self.verbose:
			print(" - %d images in %s" % (len(predictions), PredictionsDirPath))
			print(" - %d images in %s" % (len(groundtruths), GroundTruthsDirPath))
			sys.stdout.flush()

		nbImages = len(predictions)
		if nbImages != len(groundtruths):
			raise Exception("Different number of images in the directories.")
		
		
		sumAcc = sumSens = sumSpec = sumPrec = sumDice = 0.0
		evaluations = []
		for i in range(nbImages):
			pred = ImagesIO.LoadImagesList(predictions[i:i+1], True, False, False)
			gt = ImagesIO.LoadImagesList(groundtruths[i:i+1], True, False, False)
			
			dash = "" if SaveEvalutationImagesIn is None or SaveEvalutationImagesIn.endswith("/") else " - "
			path = None if SaveEvalutationImagesIn is None else SaveEvalutationImagesIn + dash + predictionsnames[i]
			
			Accuracy, Sensitivity, Specificity, Precision, Dice = Segmentation(pred[0], gt[0], Threshold=inThresh, SaveImageAs=path)
			
			if SaveEvalutationsAs is not None:
				evaluations.append({"Image": predictions[i],
								   "Accuracy": Accuracy,
								   "Sensitivity/Recall": Sensitivity,
								   "Specificity": Specificity,
								   "Precision": Precision,
								   "Dice": Dice})

				sumAcc += Accuracy
				sumSens += Sensitivity
				sumSpec += Specificity
				sumPrec += Precision
				sumDice += Dice

		if SaveEvalutationsAs is not None:
			evaluations.append({"Image": "Average",
							   "Accuracy": sumAcc / nbImages,
							   "Sensitivity/Recall": sumSens / nbImages,
							   "Specificity": sumSpec / nbImages,
							   "Precision": sumPrec / nbImages,
							   "Dice": sumDice / nbImages})
							   
			results = pandas.DataFrame(evaluations, columns=["Image", "Accuracy", "Sensitivity/Recall", "Specificity", "Precision", "Dice"])
			results.set_index("Image", inplace=True)
			results.to_csv(SaveEvalutationsAs)







def Segmentation(Prediction, GroundTruth, Threshold: int=128, SaveImageAs: str=None):
	""" This function evaluates the segmentation result.
		Args:
			Prediction: The predicted/output image.
			GroundTruth: The ground truth image.
			Threshold: The threshold to determine positive vs negative areaa.
			SaveImageAs (str): The entire path and name to save the result image. The result image is generate only if this path is not None.
		Returns:
			Accuracy
			Sensitivity/Recall
			Specificity
			Precision
			Dice
	"""

	binPred = numpy.where(Threshold < Prediction, 1, 0)
	invPred = numpy.where(Threshold < Prediction, 0, 1)

	binGT = numpy.where(Threshold < GroundTruth, 1, 0)
	invGT = numpy.where(Threshold < GroundTruth, 0, 1)

	imTP = numpy.logical_and(binPred, binGT)
	imTN = numpy.logical_and(invPred, invGT)

	imFP = numpy.logical_and(binPred, invGT)
	imFN = numpy.logical_and(invPred, binGT)

	TP = numpy.sum(imTP)
	TN = numpy.sum(imTN)
	FP = numpy.sum(imFP)
	FN = numpy.sum(imFN)

	Acc = (TP+TN) / (TP+TN+FP+FN)
	Sensitivity = Recall = float('nan') if TP+FN == 0 else TP / (TP+FN)
	Specificity = float('nan') if TN+FP == 0 else TN / (TN+FP)
	Precision = float('nan') if TP+FP == 0 else TP / (TP+FP)
	
	Union = numpy.sum(numpy.logical_or(binPred, binGT))
	Dice = float('nan') if Union == 0 else TP / Union

	if SaveImageAs is not None:
		Width, Height, Channels, First = ImageTools.Dimensions(GroundTruth)
		imres = numpy.ndarray(shape=(3, Height, Width), dtype=numpy.uint8)
		imres[0] = imres[1] = imres[2] = imTP # True positive in white
		imres[0] += imFP[0] # False positive in red
		imres[1] += imFN[0] # False negative in green
		imres = numpy.where(1 <= imres, 1, 0) * 255
		ImagesIO.Write(imres, True, SaveImageAs)
	
	
	return Acc, Sensitivity, Specificity, Precision, Dice



#import numba
#@numba.njit
def _FillVotesFirst(array1, array2, votes, sizes1, sizes2, width: int, height: int):
	for y in range(height):
		for x in range(width):
			votes[array1[:,y,x], array2[:,y,x]] += 1
			sizes1[array1[:,y,x]] += 1
			sizes2[array2[:,y,x]] += 1

def _FillVotesLast(array1, array2, votes, sizes1, sizes2, width: int, height: int):
	for y in range(height):
		for x in range(width):
			votes[array1[y,x,:], array2[y,x,:]] += 1
			sizes1[array1[y,x,:]] += 1
			sizes2[array2[y,x,:]] += 1

@timed
def Detection(Prediction, GroundTruth, Overlap: float=0.51, SaveImageAs: str=None):
	""" This function evaluates the detection results. The labeling does not matter, so labels numbers do not have to match.
		Args:
			Prediction: The predicted labels.
			GroundTruth: The ground truth labels.
			Overlap (float32): What is the minimum overlap to consider that a pattern was detected?
			SaveImageAs (str): The entire path and name to save the result image. The result image is generate only if this path is not None.
		Returns:
			Accuracy
			Sensitivity/Recall
			Specificity
			Precision
			Dice
	"""
	width, height, channel, first = ImageTools.Dimensions(Prediction)
	
	imPred = Prediction.astype(numpy.uint16)
	imGT   = GroundTruth.astype(numpy.uint16)
	nbPred = imPred.max() + 1
	nbGT   = imGT.max() + 1
	
	votes = numpy.ndarray(shape=(nbGT, nbPred), dtype=numpy.uint16)
	votes.fill(0)
	
	sizesGT = numpy.ndarray(shape=(nbGT), dtype=numpy.uint16)
	sizesGT.fill(0)
	
	sizesPred = numpy.ndarray(shape=(nbPred), dtype=numpy.uint16)
	sizesPred.fill(0)
	
	#with Timing(name='Filling votes: '):
	if first: _FillVotesFirst(imGT, imPred, votes, sizesGT, sizesPred, width, height)
	else: _FillVotesLast(imGT, imPred, votes, sizesGT, sizesPred, width, height)

	#sys.exit(0)
	TP = TN = FP = FN = 0
	FNs = numpy.ndarray(shape=(nbGT), dtype=bool)
	FNs.fill(False)
	TPs = numpy.ndarray(shape=(nbGT), dtype=bool)
	TPs.fill(False)
	#with Timing(name='Finding along nbGT: '):
	for y in range(1, nbGT):
		if 0 < sizesGT[y]:
			index = votes[y].argmax()
			if index == 0 or float(votes[y][index]) / float(sizesGT[y]) < Overlap:
				FN += 1
				FNs[y] = True
			else:
				TP += 1
				TPs[y] = True
	
	FPs = numpy.ndarray(shape=(nbPred), dtype=bool)
	FPs.fill(False)
	#with Timing(name='Finding along nbPred: '):
	for x in range(1, nbPred):
		if 0 < sizesPred[x]:
			index = votes[:, x].argmax()
			if index == 0 or float(votes[index][x]) / float(sizesPred[x]) < Overlap:
				FP += 1
				FPs[x] = True

	#print("TP="+str(TP)+", TN="+str(TN)+", FP="+str(FP)+", FN="+str(FN))
	Acc = (TP+TN) / (TP+TN+FP+FN)
	Sensitivity = Recall = float('nan') if TP+FN == 0 else TP / (TP+FN)
	Precision = float('nan') if TP+FP == 0 else TP / (TP+FP)
	
	if SaveImageAs is not None:
		imres = numpy.ndarray(shape=(3, height, width), dtype=numpy.uint8)
		if first:
			for y in range(height):
				for x in range(width):
					if TPs[imGT[:,y,x]] == True:
						imres[:,y,x] = 255
					elif FNs[imGT[:,y,x]] == True:
						imres[1,y,x] = 255
					elif FPs[imPred[:,y,x]] == True:
						imres[0,y,x] = 255
		else:
			for y in range(height):
				for x in range(width):
					if TPs[imGT[y,x,:]] == True:
						imres[:,y,x] = 255
					elif FNs[imGT[y,x,:]] == True:
						imres[1,y,x] = 255
					elif FPs[imPred[y,x,:]] == True:
						imres[0,y,x] = 255
		ImagesIO.Write(imres, True, SaveImageAs)
	

	return Acc, Sensitivity, Precision
