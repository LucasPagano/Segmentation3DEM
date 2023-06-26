import imageio

import numpy as np

import ImagesIO




def IoU(Prediction, GroundTruth, Threshold: float=0.5):
	""" This function computes the mean IoU for all classes.  It automatically computes the number of classes from the inputs.
		Args:
			Prediction:
			GroundTruth:
			Threshold:
		Return:
			The mean IoU.
	"""
	Prediction, GroundTruth = Prediction.flatten(), GroundTruth.flatten()
	if len(Prediction) != len(GroundTruth):
		print("Exception: " + str(Prediction.shape) + " => " + str(GroundTruth.shape))
		raise Exception("len(Prediction) != len(GroundTruth)")
	
	Prediction = Prediction >= Threshold
	GroundTruth = GroundTruth >= Threshold
	classes = list(np.unique(GroundTruth).astype('int8'))
						
	iou = 0
	for i in classes:
		TP = np.sum(np.logical_and(GroundTruth == i, Prediction == i) * 1).astype('float64')
		FP = np.sum(np.logical_and(GroundTruth != i, Prediction == i) * 1).astype('float64')
		FN = np.sum(np.logical_and(GroundTruth == i, Prediction != i) * 1).astype('float64')
		Intersection = TP
		Union = (TP + FP + FN)
		iou += (Intersection / Union)
	return np.sum(iou) / len(classes)


def IoU_Class(Prediction, GroundTruth, Class: int=1, Threshold: float=0.5):
	Prediction  = Prediction.flatten()
	GroundTruth = GroundTruth.flatten()
	if len(Prediction) != len(GroundTruth):
		print("Exception: " + str(Prediction.shape) + " => " + str(GroundTruth.shape))
		raise Exception("len(Prediction) != len(GroundTruth)")
	Prediction = Prediction >= Threshold
	GroundTruth = GroundTruth >= Threshold
	TP = np.sum(np.logical_and(GroundTruth == Class, Prediction == Class) * 1).astype('float64')
	FP = np.sum(np.logical_and(GroundTruth != Class, Prediction == Class) * 1).astype('float64')
	FN = np.sum(np.logical_and(GroundTruth == Class, Prediction != Class) * 1).astype('float64')
	Intersection = int(TP)
	Union = int(TP + FP + FN)
	try:
		iou = Intersection / Union
	except ZeroDivisionError:
		return 0.0
	return iou


def Accuracy(Prediction, GroundTruth, Threshold: float=0.5):
	""" This function computes the accuracy.
		Args:
			Prediction:
			GroundTruth:
			Threshold:
		Return:
			The Accuracy.
	"""
	Prediction, GroundTruth = Prediction.flatten(), GroundTruth.flatten()
	if len(Prediction) != len(GroundTruth):
		print(str(Prediction.shape) + " => " + str(GroundTruth.shape))
		raise Exception("len(Prediction) != len(GroundTruth)")
	Prediction = Prediction >= Threshold
	GroundTruth = GroundTruth >= Threshold
	return np.sum(Prediction == GroundTruth) / len(Prediction)






def Evaluate(PredictionDir: str, GroundTruthDir: str, Threshold: float=128.0, ResultFile: str=None):
	""" This function computes the accuracy and iou for an entire dataset.
		Args:
			PredictionDir (str):
			GroundTruthDir (str):
			ResultFile:
		Return:
			The lists of accuracies and ious if ResultFile is not None.
	"""
	predim = ImagesIO.FindImages(PredictionDir)
	gtim = ImagesIO.FindImages(GroundTruthDir)
	if len(predim) != len(gtim):
		raise Exception("Not the same number of images.")

	acc = []
	iou = []
	iou1 = []
	for YFile, YpredFile in zip(gtim, predim):
		Y = imageio.imread(YFile).astype(float)
		Ypred = imageio.imread(YpredFile).astype(float)
		acc.append(Accuracy(Ypred, Y, Threshold=Threshold))
		iou.append(IoU(Ypred, Y, Threshold=Threshold))
		iou1.append(IoU_Class(Ypred, Y, Threshold=Threshold))
	
	if ResultFile is not None:
		with open(ResultFile, "w") as f:
			f.write("\"Accuracy\" \"Global IoU\" \"Positive Class IoU\"\n")
			for a, i, c in zip(acc, iou, iou1):
				f.write(str(a) + " " + str(i) + " " + str(c) + "\n")
			f.write("\nAverage => " + str(np.mean(acc)) + " " + str(np.mean(iou)) + " " + str(np.mean(iou1)) + "\n")
			f.write("STD => " + str(np.std(acc)) + " " + str(np.std(iou)) + " " + str(np.std(iou1)) + "\n")
	else:
		return acc, iou, iou1


def MesCouillesXcode():
	print("")




