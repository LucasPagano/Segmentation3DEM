import glob
import os
import pandas
import sys
import time



Java_Path  = "java"
Java_Options = "-noverify -Xms4G -Xmx8G -classpath .:FiReTiTiLiB.jar:lib/*:."
Javac_Path = "javac"
Javac_Options = "-classpath .:FiReTiTiLiB.jar:lib/*:."

Python_Path = "python"




# ----------------------------------------------------- Cyclic IF -----------------------------------------------------
def CyclicIF(Parameters: dict):
	""" This function calls the cyclic IF segmentation pipeline using the given parameters.
		Args:
			Parameters (dict): A dictionary containing all the parameters.
	"""
	
	Keys = ['nbCPU', 'Experiment', 'Improve_Dapi', 'Preprocessing', 'BatchSize', 'Segment_Nuclei', 'Segment_Cells',
			'Background_Subtraction', 'Images_Directory', 'Segmentation_Directory', 'FE_SaveImages', 'FE_BiasedFeaturesOnly', 'FE_DistanceFromBorder',
			'Features_Directory']
	SegmentationKeys = Keys[0:10]
	FeaturesKeys = Keys[10:13]
	
	for key in Parameters.keys():
		if key not in Keys:
			raise Exception("Unknown parameter: '" + key + "'")
	
	
	java = open("Cluster_CyclicIF.java","w")
	
	java.write('import softwares.ohsu.cyclicif.CyclicIF;\n')
	java.write('import init.Initializer;\n\n')
	
	java.write('public class Cluster_CyclicIF\n')
	java.write('{\n\n')
	
	java.write('public static void main(String[] args) throws Exception\n')
	java.write('\t{\n')
	java.write('\tInitializer.Start() ;\n\n')
	
	if 'Images_Directory' not in Parameters: raise Exception("Images_Directory parameter missing.")
	
	if 'nbCPU' not in Parameters: raise Exception("nbCPU parameter missing.")
	java.write('\tfinal int nbCPU = ' + str(Parameters['nbCPU']) + ' ;\n\n')
	
	
	java.write('\tCyclicIF cif = new CyclicIF(20) ;\n\n')
	java.write('\tcif.Python_Path = "' + Python_Path + '" ;\n\n')
	
	if 'Segmentation_Directory' in Parameters and Parameters['Segmentation_Directory'] is not None and Parameters['Segmentation_Directory']:
		
		if 'Experiment' not in Parameters: raise Exception("Experiment type missing.")
		java.write('\tcif.Experiment = cif.' + Parameters['Experiment'] + ' ;\n')
		
		java.write('\tcif.Improve_Dapi = ')
		if 'Improve_Dapi' not in Parameters or Parameters['Improve_Dapi'] == None or Parameters['Improve_Dapi'] == False: java.write('false ;\n')
		elif Parameters['Improve_Dapi'] == True: java.write('true ;\n')
		else: raise Exception("Wrong value for parameter 'Improve_Dapi', True or False expected, '" + Parameters['Improve_Dapi'] + "' found.")
			
		java.write('\tcif.Preprocessing = ')
		if 'Preprocessing' not in Parameters or Parameters['Preprocessing'] == None or Parameters['Preprocessing'] == False: java.write('false ;\n')
		elif Parameters['Preprocessing'] == True: java.write('true ;\n')
		else: raise Exception("Wrong value for parameter 'Preprocessing', True or False expected, '" + Parameters['Preprocessing'] + "' found.")
		
		java.write('\tcif.Segment_Nuclei = ')
		if 'Segment_Nuclei' not in Parameters or Parameters['Segment_Nuclei'] == None: java.write('null ;\n')
		else: java.write('"' + Parameters['Segment_Nuclei'] + '" ;\n')
		
		java.write('\tcif.Segment_Cells = ')
		if 'Segment_Cells' not in Parameters or Parameters['Segment_Cells'] == None or Parameters['Segment_Cells'] == False: java.write('false ;\n')
		elif Parameters['Segment_Cells'] == True: java.write('true ;\n')
		else: raise Exception("Wrong value for parameter 'Segment_Cells', True or False expected, '" + Parameters['Segment_Cells'] + "' found.")
		
		java.write('\tcif.Background_Subtraction = ')
		if 'Background_Subtraction' not in Parameters or Parameters['Background_Subtraction'] == None or Parameters['Background_Subtraction'] == False:
			java.write('false ;\n')
		elif Parameters['Background_Subtraction'] == True:
			java.write('true ;\n')
		else:
			raise Exception("Wrong value for parameter 'Background_Subtraction', True or False expected, '"
							+ Parameters['Background_Subtraction'] + "' found.")
		
		java.write('\tcif.SegmentNuclei_BatchSize = ')
		if 'BatchSize' not in Parameters or Parameters['BatchSize'] == None: java.write('3 ;\n')
		elif isinstance(Parameters['BatchSize'], int) and 0 < Parameters['BatchSize']: java.write('%d ;\n' % Parameters['BatchSize'])
		else: raise Exception("Wrong value for parameter 'BatchSize', positive integer expected, '" + Parameters['BatchSize'] + "' found.")
		
		java.write('\tcif.Segment("' + Parameters['Images_Directory'] + '", "' + Parameters['Segmentation_Directory'] + '", nbCPU) ;\n\n')
	
	
	
		if 'Features_Directory' in Parameters and Parameters['Features_Directory'] is not None and Parameters['Features_Directory']:
		
			java.write('\tcif.FE_SaveImages = ')
			if 'FE_SaveImages' not in Parameters or Parameters['FE_SaveImages'] == None or Parameters['FE_SaveImages'] == False: java.write('false ;\n')
			elif Parameters['FE_SaveImages'] == True: java.write('true ;\n')
			else: raise Exception("Wrong value for parameter 'FE_SaveImages', True or False expected, '" + Parameters['FE_SaveImages'] + "' found.")
			
			java.write('\tcif.FE_BiasedFeaturesOnly = ')
			if 'FE_BiasedFeaturesOnly' not in Parameters or Parameters['FE_BiasedFeaturesOnly'] == None or Parameters['FE_BiasedFeaturesOnly'] == False:
				java.write('false ;\n')
			elif Parameters['FE_BiasedFeaturesOnly'] == True:
				java.write('true ;\n')
			else:
				raise Exception("Wrong value for parameter 'FE_BiasedFeaturesOnly', True or False expected, '"
								+ Parameters['FE_BiasedFeaturesOnly'] + "' found.")
			
			java.write('\tcif.FE_DistanceFromBorder = ')
			if 'FE_DistanceFromBorder' not in Parameters or Parameters['FE_DistanceFromBorder'] == None or Parameters['FE_DistanceFromBorder'] == False:
				java.write('false ;\n')
			elif Parameters['FE_DistanceFromBorder'] == True:
				java.write('true ;\n')
			else:
				raise Exception("Wrong value for parameter 'FE_DistanceFromBorder', True or False expected, '"
								+ Parameters['FE_DistanceFromBorder'] + "' found.")
			
			java.write('\tcif.FeaturesExtraction("' + Parameters['Images_Directory'] + '", "'
													+ Parameters['Segmentation_Directory'] + '", "'
													+ Parameters['Features_Directory'] + '", nbCPU) ;\n\n')
			
	java.write('\tSystem.exit(0) ;\n')
	java.write('\t}\n')
	java.write('}\n')
	java.close()

	
	exit = os.system(Javac_Path + " " + Javac_Options + " Cluster_CyclicIF.java")
	print("Java compilation done and exited with status " + str(exit) + "\n")
	
	exit = os.system(Java_Path + " " + Java_Options + " Cluster_CyclicIF")
	print("Processing done and exited with status " + str(exit))
	
	













# ----------------------------------------------------- Evaluate Detection -----------------------------------------------------
def EvaluateDetection(Prediction: str, GroundTruth: str, Overlap: float, SaveImageAs: str):
	""" This function calls the measures.Evaluations.Detection method.
		Args:
			Prediction (str): .
			GroundTruth (str): .
			Overlap (float): .
			SaveImageAs (str): .
	"""
	
	java = open("EvaluateDetection.java","w")
	
	java.write('import measures.Evaluations;\n')
	java.write('import init.Initializer;\n\n')
	
	java.write('public class EvaluateDetection\n')
	java.write('{\n\n')
	
	java.write('public static void main(String[] args) throws Exception\n')
	java.write('\t{\n')
	java.write('\tInitializer.Start() ;\n')
	
	java.write('\tEvaluations.Detection("')
	java.write(Prediction)
	java.write('", "')
	java.write(GroundTruth)
	java.write('", (float)')
	java.write(str(Overlap))
	java.write(', "')
	java.write(SaveImageAs)
	java.write('", "tmp.csv") ;\n')
	
	java.write('\tSystem.exit(0) ;\n')
	java.write('\t}\n')
	java.write('}\n')
	java.close()

	exit = os.system(Javac_Path + " " + Javac_Options + " EvaluateDetection.java")
	print("Java compilation done and exited with status " + str(exit) + "\n")
	
	exit = os.system(Java_Path + " " + Java_Options + " EvaluateDetection")
	print("Processing done and exited with status " + str(exit))
	
	file = pandas.read_csv("tmp.csv")
	line = file.to_numpy()
	
	os.remove("EvaluateDetection.java")
	os.remove("EvaluateDetection.class")
	os.remove("tmp.csv")
	
	return line[0,0], line[0,1], line[0,2]
	

