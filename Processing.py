import os
import sys



def ForceRange(x, min: float=0.0, max: float=255.0):
	""" This function cuts excessive values. As a consequence, the final values are into the range [min, max].
		Args:
			x: The data to shrink.
			min (:obj:`float`, optional): The minimum possible value. Defaults to 0.0.
			max (:obj:`float`, optional): The maximum possible value. Defaults to 255.0.
	"""
	x[x < min] = min
	x[max < x] = max
