

def Dimensions(image):
	""" This function finds the image dimensions.
		Args:
			image: The image to analyze.
		Returns:
			Width (int), Height (int), #Channels (int), Are the channels first (bool)?
	"""

	if len(image.shape) == 2: # Gray level images.
		shape = image.shape
		channels = 1
		width = shape[1]
		height = shape[0]
		first = True
	elif image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]: # channel first
		shape = image.shape
		channels = shape[0]
		width = shape[2]
		height = shape[1]
		first = True
	elif image.shape[2] < image.shape[0] and image.shape[2] < image.shape[1]: # channel last
		shape = image.shape
		channels = shape[2]
		width = shape[1]
		height = shape[0]
		first = False
	else:
		raise Exception("Dimensions not found.")

	return width, height, channels, first
