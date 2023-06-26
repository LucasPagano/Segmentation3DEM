import glob
import math
import numpy
import os
import pandas
import sys
import time

sys.path.insert(0, '../')
import Evaluations
import ImagesIO
import JavaInterfacer

from horology import timed, Timing




class MaskRCNN(object):
	
	def __init__(self, MaxLevel: int=255, verbose: bool=True):
		""" Initialization.
			Args:
				MaxLevel (int): The maximum possible value in the images. Defaults to 255.
				verbose (bool, optional): Defaults to True.
		"""
		self.__Reset__()
		self.MaxLevel = MaxLevel
		self.verbose = verbose
	
	
	def __Reset__(self):
		pass


	@timed
	def Directory(self, PredictionsDirPath, GroundTruthsDirPath, Threshold: int=128, Overlap: float=0.51,
				SaveImagesIn: str=None, SaveEvalutationsAs: str=None, SegmentationOnly: bool=False):
		""" This function evaluates the segmentation and detection results for all the images in a given directory.
			Args:
				PredictionsDirPath: The path of the directory containing the results.
				GroundTruthsDirPath: The path of the directory containing the ground truth (labels) images.
				Threshold (int): The threshold to determine positive vs negative areaa for the segmentation.
				Overlap (float32): What is the minimum overlap to consider that a pattern was detected?
				SaveImagesIn (str): The path of the directory where the results will be saved.
				SaveEvalutationsAs (str):
				SegmentationOnly (bool): Evaluate only the foreground/segmentation, not the detection?
			Returns:
				Segmentation Accuracy
				Segmentation Sensitivity/Recall
				Segmentation Specificity
				Segmentation Precision
				Segmentation Dice
				Detection Accuracy (if SegmentationOnly is false)
				Detection Sensitivity/Recall (if SegmentationOnly is false)
				Detection Precision (if SegmentationOnly is false)
		"""

		if SaveImagesIn is not None:
			os.makedirs(SaveImagesIn, exist_ok=True)
		
		DetectionsPaths = self._FindImages(PredictionsDirPath, "- Labels")
		ForegroundPaths = self._FindImages(PredictionsDirPath, "- Foreground")
		
		GT = ImagesIO.FindImages(GroundTruthsDirPath, NamesOnly=False, verbose=False)
		
		if self.verbose:
			print(" - %d labels in %s" % (len(DetectionsPaths), PredictionsDirPath))
			print(" - %d foregrounds in %s" % (len(ForegroundPaths), PredictionsDirPath))
			print(" - %d images in %s" % (len(GT), GroundTruthsDirPath))
			sys.stdout.flush()
		
		nbImages = len(GT)
		if len(DetectionsPaths) != nbImages or len(ForegroundPaths) != nbImages:
			raise Exception("Different number of images.")
		
		SegSumAcc = SegSumSens = SegSumSpec = SegSumPrec = SegSumDice = 0.0
		DetSumAcc = DetSumSens = DetSumPrec = 0.0
		evaluations = []
		for i in range(nbImages):
			prefix = os.path.commonprefix([DetectionsPaths[i], ForegroundPaths[i]])
			name = os.path.basename(prefix)
			dash = "" if SaveImagesIn is None or SaveImagesIn.endswith("/") else " - "
			path = None if SaveImagesIn is None else SaveImagesIn + dash + name
			print("Evaluating '" + prefix + "'.")
			
			if SegmentationOnly == False: det = ImagesIO.Read(DetectionsPaths[i], True, False)
			seg = ImagesIO.Read(ForegroundPaths[i], True, False)
			gt = ImagesIO.Read(GT[i], True, False)
			segGT = numpy.where(0 < gt, 255, 0)
			
			if SegmentationOnly == False:
				SegAcc, SegSens, SegSpec, SegPrec, SegDice, DetAcc, DetSens, DetPrec = self.EvaluateImage(seg, segGT, det, DetectionsPaths[i], gt, GT[i],
														Threshold=Threshold, Overlap=Overlap, SaveImagesAs=path, SegmentationOnly=SegmentationOnly)
			else:
				SegAcc, SegSens, SegSpec, SegPrec, SegDice = self.EvaluateImage(seg, segGT, None, None, gt, None,
														Threshold=Threshold, Overlap=Overlap, SaveImagesAs=path, SegmentationOnly=SegmentationOnly)
			
			if SaveEvalutationsAs is not None:
				if SegmentationOnly == False:
					evaluations.append({"Image": name,
									   "Segmentation_Accuracy": SegAcc,
									   "Segmentation_Sensitivity/Recall": SegSens,
									   "Segmentation_Specificity": SegSpec,
									   "Segmentation_Precision": SegPrec,
									   "Segmentation_Dice": SegDice,
									   "Detection_Accuracy": DetAcc,
									   "Detection_Sensitivity/Recall": DetSens,
									   "Detection_Precision": DetPrec})
				else:
					evaluations.append({"Image": name,
										"Segmentation_Accuracy": SegAcc,
										"Segmentation_Sensitivity/Recall": SegSens,
										"Segmentation_Specificity": SegSpec,
										"Segmentation_Precision": SegPrec,
										"Segmentation_Dice": SegDice})
				SegSumAcc  += SegAcc
				SegSumSens += SegSens
				SegSumSpec += SegSpec
				SegSumPrec += SegPrec
				SegSumDice += SegDice
				if SegmentationOnly == False:
					DetSumAcc  += DetAcc
					DetSumSens += DetSens
					DetSumPrec += DetPrec

		
		if SaveEvalutationsAs is not None:
			if SegmentationOnly == False:
				evaluations.append({"Image": "Average",
									"Segmentation_Accuracy": SegSumAcc / nbImages,
									"Segmentation_Sensitivity/Recall": SegSumSens / nbImages,
									"Segmentation_Specificity": SegSumSpec / nbImages,
									"Segmentation_Precision": SegSumPrec / nbImages,
									"Segmentation_Dice": SegSumDice / nbImages,
									"Detection_Accuracy": DetSumAcc / nbImages,
									"Detection_Sensitivity/Recall": DetSumSens / nbImages,
									"Detection_Precision": DetSumPrec / nbImages})
			else:
				evaluations.append({"Image": "Average",
									"Segmentation_Accuracy": SegSumAcc / nbImages,
									"Segmentation_Sensitivity/Recall": SegSumSens / nbImages,
									"Segmentation_Specificity": SegSumSpec / nbImages,
									"Segmentation_Precision": SegSumPrec / nbImages,
									"Segmentation_Dice": SegSumDice / nbImages})
			
			results = pandas.DataFrame(evaluations)
			results.set_index("Image", inplace=True)
			
			if SegmentationOnly == False:
				results.rename(columns={"Segmentation_Accuracy" : "Segmentation Accuracy",
										"Segmentation_Sensitivity/Recall" : "Segmentation Sensitivity/Recall",
										"Segmentation_Specificity" : "Segmentation Specificity",
										"Segmentation_Precision" : "Segmentation Precision",
										"Segmentation_Dice" : "Segmentation Dice",
										"Detection_Accuracy" : "Detection Accuracy",
										"Detection_Sensitivity/Recall" : "Detection Sensitivity/Recall",
										"Detection_Precision" : "Detection Precision"},
										inplace=True)
			else:
				results.rename(columns={"Segmentation_Accuracy" : "Segmentation Accuracy",
										"Segmentation_Sensitivity/Recall" : "Segmentation Sensitivity/Recall",
										"Segmentation_Specificity" : "Segmentation Specificity",
										"Segmentation_Precision" : "Segmentation Precision",
										"Segmentation_Dice" : "Segmentation Dice"},
										inplace=True)
										
			acc  = results.pop("Segmentation Accuracy")
			dice = results.pop("Segmentation Dice")
			results.insert(0, "Segmentation Dice", dice)
			results.insert(0, "Segmentation Accuracy", acc)
			results.to_csv(SaveEvalutationsAs)
			
	
	
	def _FindImages(self, DirPath: str, Key: str) -> list:
		""" This function fins all the images in the given directory.
			Args:
				DirPath (str): The directory's path containing the images.
				Key (sr): .
			Returns:
				A list of all the image paths in the given directory.
		"""
		list = glob.glob(DirPath + "/*" + Key + "*.*")
		list.sort()
		return list
	
	
	def EvaluateImage(self, SegPred, SegGT, DetPred, DetPredPath, DetGT, DetGTPath, Threshold: int=128, Overlap: float=0.51, SaveImagesAs: str=None,
					SegmentationOnly: bool=False):
		""" This function evaluates the segmentation and prediction results.
			Args:
				SegPred: The predicted/output/segmented image.
				SegGT: The segmentation ground truth image.
				DegPred: The predicted/output/detections image.
				DetGT: The detections ground truth image.
				Threshold (int): The threshold to determine positive vs negative areaa for the segmentation.
				Overlap (float32): What is the minimum overlap to consider that a pattern was detected?
				SaveImageAs (str): The entire path and prefix name to save the result images. If None, no result images saved.
			Returns:
				Segmentation returns, Detection returns
		"""
		SaveSegmentationImageAs = None if SaveImagesAs is None else SaveImagesAs + " - Segmentation.png"
		SaveDetectionImageAs = None if SaveImagesAs is None else SaveImagesAs + " - Detection.png"
		
		Acc, Sensitivity, Specificity, Precision, Dice = self.Segmentation(SegPred, SegGT, Threshold=Threshold, SaveImageAs=SaveSegmentationImageAs)
		
		if SegmentationOnly == False:
			#Acc2, Sensitivity2, Precision2 = self.Detection(DetPred, DetGT, Overlap=Overlap, SaveImageAs=SaveDetectionImageAs)
			Acc2, Sensitivity2, Precision2 = self.Detection2(DetPredPath, DetGTPath, Overlap=Overlap, SaveImageAs=SaveDetectionImageAs)
			return Acc, Sensitivity, Specificity, Precision, Dice, Acc2, Sensitivity2, Precision2
		
		return Acc, Sensitivity, Specificity, Precision, Dice
		



	def Segmentation(self, Prediction, GroundTruth, Threshold: int=128, SaveImageAs: str=None):
		""" This function evaluates the segmentation result for a single given image.
			Args:
				Prediction: The predicted/output image.
				GroundTruth: The ground truth image.
				Threshold: The threshold to determine positive vs negative areaa.
				SaveImageAs (str): The entire path and name to save the result image. The result image is generated only if this path is not None.
			Returns:
				Accuracy
				Sensitivity/Recall
				Specificity
				Precision
				Dice
		"""
		return Evaluations.Segmentation(Prediction, GroundTruth, Threshold, SaveImageAs)
	
	
	
	def Detection(self, Prediction, GroundTruth, Overlap: float=0.51, SaveImageAs: str=None):
		""" This function evaluates the detection results for a single given image.
			Args:
				Prediction: The predicted labels.
				GroundTruth: The ground truth labels.
				Overlap (float32): What is the minimum overlap to consider that a pattern was detected?
				SaveImageAs (str): The entire path and name to save the result image. The result image is generated only if this path is not None.
			Returns:
				Accuracy
				Sensitivity/Recall
				Precision
		"""
		return Evaluations.Detection(Prediction, GroundTruth, Overlap, SaveImageAs)
	
	def Detection2(self, Prediction, GroundTruth, Overlap: float=0.51, SaveImageAs: str=None):
		""" This function evaluates the detection results for a single given image.
			Args:
				Prediction: The predicted labels.
				GroundTruth: The ground truth labels.
				Overlap (float32): What is the minimum overlap to consider that a pattern was detected?
				SaveImageAs (str): The entire path and name to save the result image. The result image is generated only if this path is not None.
			Returns:
				Accuracy
				Sensitivity/Recall
				Precision
		"""
		return JavaInterfacer.EvaluateDetection(Prediction, GroundTruth, Overlap, SaveImageAs)




