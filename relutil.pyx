# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np

cdef BilinearSample(np.ndarray[np.uint8_t, ndim=3] imgPix, 
	pilImg,
	float x, float y, 
	np.ndarray[np.float64_t, ndim=2] p, #Temporary storage
	np.ndarray[np.float64_t, ndim=2] out,
	int row):

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

	for c in range(imgPix.shape[2]):
		c1 = p[0,c] * (1.-xfrac) + p[1,c] * xfrac
		c2 = p[2,c] * (1.-xfrac) + p[3,c] * xfrac
		out[row,c] = c1 * (1.-yfrac) + c2 * yfrac

cdef BilinearSampleOld(imgPix, float x, float y):
	cdef float xfrac, xi, yfrac, yi

	xfrac, xi = math.modf(x)
	yfrac, yi = math.modf(y)

	#Get surrounding pixels
	p00 = imgPix[yi,xi,:]
	p10 = imgPix[yi,xi+1,:]
	p01 = imgPix[yi+1,xi,:]
	p11 = imgPix[yi+1,xi+1,:]

	#If a single number has been returned, convert to list
	if not hasattr(p00, '__iter__'): p00 = [p00]
	if not hasattr(p10, '__iter__'): p10 = [p10]
	if not hasattr(p01, '__iter__'): p01 = [p01]
	if not hasattr(p11, '__iter__'): p11 = [p11]

	#Interpolate colour
	c1 = [p00c * (1.-xfrac) + p10c * xfrac for p00c, p10c in zip(p00, p10)]
	c2 = [p01c * (1.-xfrac) + p11c * xfrac for p01c, p11c in zip(p01, p11)]
	col = [c1c * (1.-yfrac) + c2c * yfrac for c1c, c2c in zip(c1, c2)]

	return col


def GetPixIntensityAtLoc(np.ndarray[np.uint8_t, ndim=3] iml, 
	pilImg,
	np.ndarray[np.float64_t, ndim=2] supportOffsets, 
	float locx, float locy,		
	float rotation = 0.):

	cdef np.ndarray[np.float64_t, ndim=2] out = np.zeros((supportOffsets.shape[0], iml.shape[2]))
	cdef double x, y
	cdef float rx, ry
	cdef float offsetX, offsetY
	cdef int offsetNum

	cdef np.ndarray[np.float64_t, ndim=2] temp = np.empty((4, iml.shape[2]))

	for offsetNum in range(supportOffsets.shape[0]):
		offsetX = supportOffsets[offsetNum, 0]
		offsetY = supportOffsets[offsetNum, 1]

		#Apply rotation (anti-clockwise)
		rx = math.cos(rotation) * offsetX - math.sin(rotation) * offsetY
		ry = math.sin(rotation) * offsetX + math.cos(rotation) * offsetY

		#Get pixel at this location
		try:
			x = rx + locx
			y = ry + locy
			BilinearSample(iml, pilImg, x, y, temp, out, offsetNum)
			#oldCol = BilinearSampleOld(iml, x, y)
			#print out[offsetNum, :], oldCol
			#out[offsetNum, :] = np.array(oldCol)
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
