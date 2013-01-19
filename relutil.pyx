# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math

cdef BilinearSample(imgPix, float x, float y):
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
