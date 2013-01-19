# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np

cdef BilinearSample(imgPix, float x, float y):

	cdef np.ndarray[np.uint8_t, ndim=2] p = np.empty((4, imgPix.shape[2]), dtype=np.uint8)
	cdef np.ndarray out = np.empty((imgPix.shape[2]))
	cdef int c
	cdef int xi = int(x)
	cdef double xfrac = x - xi
	cdef int yi = int(y)
	cdef double yfrac = y - yi
	cdef double c1, c2

	#Check bounds
	if xi < 0 or xi + 1 >= imgPix.shape[1]:
		raise IndexError("Pixel location outside image")
	if yi < 0 or yi + 1 >= imgPix.shape[0]:
		raise IndexError("Pixel location outside image")

	#Get surrounding pixels
	for c in range(imgPix.shape[2]):
		p[0,c] = imgPix[yi, xi, c]
		p[1,c] = imgPix[yi, xi+1, c]
		p[2,c] = imgPix[yi+1, xi, c]
		p[3,c] = imgPix[yi+1, xi+1, c]

	#Interpolate colour
	for i in range(imgPix.shape[2]):
		c1 = p[0,c] * (1.-xfrac) + p[1,c] * xfrac
		c2 = p[2,c] * (1.-xfrac) + p[3,c] * xfrac
		out[c] = c1 * (1.-yfrac) + c2 * yfrac

	return out

def GetPixIntensityAtLoc(iml, supportOffsets, float locx, float locy, float rotation = 0.):
	cdef float x, y, rx, ry
	cdef float offsetX, offsetY

	out = []
	for offset in supportOffsets:
		offsetX = offset[0]
		offsetY = offset[1]

		#Apply rotation (anti-clockwise)
		rx = math.cos(rotation) * offsetX - math.sin(rotation) * offsetY
		ry = math.sin(rotation) * offsetX + math.cos(rotation) * offsetY

		#Get pixel at this location
		try:
			x = rx + locx
			y = ry + locy
			out.append(BilinearSample(iml, x, y))
		except IndexError:
			return None
	return out

def ITUR6012(col): #ITU-R 601-2
	return 0.299*col[0] + 0.587*col[1] + 0.114*col[2]

def ToGrey(col):
	if not hasattr(col, '__iter__'): return col
	if len(col) == 3:
		return ITUR6012(col)
	#Assumed to be already grey scale
	return col[0]
