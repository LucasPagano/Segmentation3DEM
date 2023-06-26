import numpy
import os
import sys

from Processing import ForceRange




class Normalize(object):
	""" This class performs a classic normalization/denormalization: [0,max] <=> [-1,1].
	"""
	
	def __init__(self, MaxValue: float=255.0):
		""" Simple initialization.
			Args:
				MaxValue (float, optional): The maximum value of the input data.
		"""
		if MaxValue <= 0.0:
			raise Exception("MaxValue <= 0.0")
		self.MaxValue = MaxValue



	def Normalize(self, x):
		""" This function performs the normalization.
			Args:
				x: The data to normalize.
		"""
		if isinstance(x, list):
			for element in x:
				self._Normalize(element)
		else:
			self._Normalize(x)

	def _Normalize(self, x):
		""" This function performs the normalization.
			Args:
				x: The data (array) to normalize.
		"""
		x /= self.MaxValue
		x -= 0.5
		x *= 2.0

	def NormalizeScalar(self, x):
		""" This function performs the normalization of a scalar.
			Args:
				x: The value to normalize.
			Returns:
				The normalized value.
		"""
		x /= self.MaxValue
		x -= 0.5
		x *= 2.0
		return x



	def Denormalize(self, x):
		""" This function performs the denormalization.
			Args:
				x: The data to denormalize.
		"""
		if isinstance(x, list):
			for element in x:
				self._Denormalize(element)
		else:
			self._Denormalize(x)

	def _Denormalize(self, x):
		""" This function performs the denormalization.
			Args:
				x: The data to denormalize.
		"""
		x /= 2.0
		x += 0.5
		x *= self.MaxValue
		ForceRange(x, min=0.0, max=self.MaxValue)

	def DenormalizeScalar(self, x):
		""" This function performs the denormalization of a scalar.
			Args:
				x: The value to denormalize.
			Returns:
				The denormalized value.
		"""
		x /= 2.0
		x += 0.5
		x *= self.MaxValue
		if x < 0.0:
			x = 0.0
		elif self.MaxValue < x:
			x = self.MaxValue
		return x



	def getMaxValue(self):
		""" This function returns the maximum value.
			Returns:
				The maximum value.
		"""
		return self.MaxValue









class Basic(object):
	""" This class performs a basic normalization/denormalization: [0,max] <=> [0,1].
	"""
	
	def __init__(self, MaxValue: float=255.0):
		""" Simple initialization.
			Args:
				MaxValue (float, optional): The maximum value of the input data.
		"""
		if MaxValue <= 0.0:
			raise Exception("MaxValue <= 0.0")
		self.MaxValue = MaxValue
	
	
	
	def Normalize(self, x):
		""" This function performs the normalization.
			Args:
				x: The data to normalize.
		"""
		if isinstance(x, list):
			for element in x:
				self._Normalize(element)
		else:
			self._Normalize(x)

	def _Normalize(self, x):
		""" This function performs the normalization.
			Args:
				x: The data (array) to normalize.
		"""
		x /= self.MaxValue

	def NormalizeScalar(self, x):
		""" This function performs the normalization of a scalar.
			Args:
				x: The value to normalize.
			Returns:
				The normalized value.
		"""
		x /= self.MaxValue
		return x



	def Denormalize(self, x):
		""" This function performs the denormalization.
			Args:
				x: The data to denormalize.
		"""
		if isinstance(x, list):
			for element in x:
				self._Denormalize(element)
		else:
			self._Denormalize(x)

	def _Denormalize(self, x):
		""" This function performs the denormalization.
			Args:
				x: The data to denormalize.
		"""
		x *= self.MaxValue
		ForceRange(x, min=0.0, max=self.MaxValue)

	def DenormalizeScalar(self, x):
		""" This function performs the denormalization of a scalar.
			Args:
				x: The value to denormalize.
			Returns:
				The denormalized value.
		"""
		x *= self.MaxValue
		if x < 0.0:
			x = 0.0
		elif self.MaxValue < x:
			x = self.MaxValue
		return x



	def getMaxValue(self):
		""" This function returns the maximum value.
			Returns:
				The maximum value.
		"""
		return self.MaxValue








class CenterReduce(object):
	""" This class centers (average 0) and reduces (standard deviation 1).
	"""
	
	def __init__(self, MaxValue: float=255.0):
		""" Simple initialization.
			Args:
				MaxValue (float, optional): The maximum value of the input data.
		"""
		if MaxValue <= 0.0:
			raise Exception("MaxValue <= 0.0")
		self.MaxValue = MaxValue
	
	
	
	def Normalize(self, x):
		""" This function performs the normalization.
			Args:
				x: The data to normalize.
		"""
		if isinstance(x, list):
			for element in x:
				self._Normalize(element)
		elif isinstance(x[0], object):
			ave = []
			std = []
			for element in x:
				self._Normalize(element)
				ave.append(self.ave)
				std.append(self.std)
			self.ave = ave
			self.std = std
		else:
			print(x.shape)
			self._Normalize(x)

	def _Normalize(self, x):
		""" This function performs the normalization.
			Args:
				x: The data (array) to normalize.
		"""
		self.ave = numpy.average(x)
		self.std = numpy.std(x)
		x -= self.ave
		x /= self.std
	
	def NormalizeScalar(self, x):
		""" This function performs the normalization of a scalar.
			Args:
				x: The value to normalize.
			Returns:
				The normalized value.
		"""
		return None #raise Exception("Method not implemented (yet).")
	


	def Denormalize(self, x):
		""" This function performs the denormalization.
			Args:
				x: The data to denormalize.
		"""
		if isinstance(x, list):
			for element in x:
				self._Denormalize(element)
		else:
			self._Denormalize(x)

	def _Denormalize(self, x):
		""" This function performs the denormalization.
			Args:
				x: The data to denormalize.
		"""
		x *= self.std
		x += self.ave
		ForceRange(x, min=0.0, max=self.MaxValue)
	
	def DenormalizeScalar(self, x):
		""" This function performs the denormalization of a scalar.
			Args:
				x: The value to denormalize.
			Returns:
				The denormalized value.
		"""
		return None #raise Exception("Method not implemented (yet).")



	def getMaxValue(self):
		""" This function returns the maximum value.
			Returns:
				The maximum value.
		"""
		return self.MaxValue





class CenterReduceGlobal(object):
	""" This class normalize data by centering (average 0) and reduceing (standard deviation 1) the entire dataset, but the average and standard
		deviation are computed on the entire dataset, not image per image.
	"""
	
	def __init__(self, MaxValue: float=255.0, RemoveOutliers: int=0):
		""" Simple initialization.
			Args:
				MaxValue (float, optional): The maximum value of the input data.
				RemoveOutliers (int, optional): The number of outliers to remove on each side before normalization.
		"""
		if MaxValue <= 0.0:
			raise Exception("MaxValue <= 0.0")
		self.MaxValue = MaxValue

		if RemoveOutliers < 0:
			raise Exception("RemoveOutliers < 0")
		self.RemoveOutliers = RemoveOutliers

		self.ave = None
		self.std = None



	def Normalize(self, x):
		""" This function performs the normalization.
			Args:
				x: The data to normalize.
		"""
		ave = []
		std = []
		if isinstance(x, list) or isinstance(x[0], object):
			for element in x:
				ave.append(numpy.average(element))
				std.append(numpy.std(element))
		else:
			raise Exception("Data format not supported (yet).")
		
		if 0 < self.RemoveOutliers:
			ave = numpy.sort(ave)
			ave = ave[self.RemoveOutliers:ave.shape[0]-self.RemoveOutliers]
			std = numpy.sort(std)
			std = std[self.RemoveOutliers:std.shape[0]-self.RemoveOutliers]
		
		self.ave = numpy.average(ave)
		self.std = numpy.average(std)
		
		for element in x:
			element -= self.ave
			element /= self.std
		

	

	def NormalizeScalar(self, x):
		""" This function performs the normalization of a scalar.
			Args:
				x: The value to normalize.
			Returns:
				The normalized value.
		"""
		return (x - self.ave) / self.std
		

		
	def Denormalize(self, x):
		""" This function performs the denormalization.
			Args:
				x: The data to denormalize.
		"""
		if isinstance(x, list) or isinstance(x[0], object):
			for element in x:
				self._Denormalize(element)
		else:
			self._Denormalize(x)

	def _Denormalize(self, x):
		""" This function performs the denormalization.
			Args:
				x: The data to denormalize.
		"""
		x *= self.std
		x += self.ave
		ForceRange(x, min=0.0, max=self.MaxValue)

	def DenormalizeScalar(self, x):
		""" This function performs the denormalization of a scalar.
			Args:
				x: The value to denormalize.
			Returns:
				The denormalized value.
		"""
		return None #raise Exception("Method not implemented (yet).")



	def getMaxValue(self):
		""" This function returns the maximum value.
			Returns:
				The maximum value.
		"""
		return self.MaxValue

	def getFeatures(self):
		""" This function returns the features used for normalization.
			Returns:
				The mean and standard deviation.
		"""
		return self.ave, self.std


