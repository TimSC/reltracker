
from PIL import Image
import time, math, pickle, sys
import numpy as np
import sklearn.ensemble as ensemble

def BilinearSample(imgPix, x, y):
	xfrac, xi = math.modf(x)
	yfrac, yi = math.modf(y)

	#Get surrounding pixels
	p00 = imgPix[xi,yi]
	p10 = imgPix[xi+1,yi]
	p01 = imgPix[xi,yi+1]
	p11 = imgPix[xi+1,yi+1]

	#Interpolate colour
	c1 = [p00c * (1.-xfrac) + p10c * xfrac for p00c, p10c in zip(p00, p10)]
	c2 = [p01c * (1.-xfrac) + p11c * xfrac for p01c, p11c in zip(p01, p11)]
	col = [c1c * (1.-yfrac) + c2c * yfrac for c1c, c2c in zip(c1, c2)]

	return col

def ITUR6012(col): #ITU-R 601-2
	return 0.299*col[0] + 0.587*col[1] + 0.114*col[2]

def ReadPosData(fina):
	data = open(fina).readlines()
	numFrames = int(data[0])
	pos = 1
	out = {}

	for frameNum in range(numFrames):
		numPts = int(data[pos])
		frameTime = int(data[pos+1])
		pts = []
		for ptNum in range(numPts):
			pt = map(float,data[pos+2+ptNum].strip().split(" "))
			pts.append(pt)
		out[frameTime] = pts

		pos += 2 + numPts
	return out

#*******************************************************************************

class RelAxis:
	def __init__(self):
		pass

	def Train(self):
		print "Train"

class RelTracker:
	def __init__(self):
		self.trainingData = []

	def Add(self, im, pos):
		self.trainingData.append((im, pos))
		assert(len(self.trainingData[0][1]) == len(self.trainingData[-1][1]))


	def Train(self):
		
		assert(len(self.trainingData)>0)
		numTrackers = len(self.trainingData[0][1])
		scalePredictors = []

		#First layer of hierarchy
		layer = []
		for trNum in range(numTrackers):
			for axis in ['x', 'y']:
				relaxis = RelAxis()
				relaxis.shapeNoise = 12
				relaxis.cloudEnabled = 1
				relaxis.supportMaxOffset = 39
				relaxis.trainVarianceOffset = 41
				relaxis.trainingData = self.trainingData
				layer.append(relaxis)
		scalePredictors.append(layer)

		#Second layer of hierarchy
		layer = []
		for trNum in range(numTrackers):
			for axis in ['x', 'y']:
				relaxis = RelAxis()
				relaxis.shapeNoise = 1
				relaxis.cloudEnabled = 0
				relaxis.supportMaxOffset = 20
				relaxis.trainVarianceOffset = 5
				relaxis.trainingData = self.trainingData
				layer.append(relaxis)
		scalePredictors.append(layer)

		
		#Train individual axis predictors
		for layer in scalePredictors:
			for relaxis in layer:
				relaxis.Train()


if __name__ == "__main__":
	posData = ReadPosData(sys.argv[1])

	reltracker = RelTracker()
	for ti in posData:
		imgFina = sys.argv[2]+"/{0:05d}.png".format(ti)
		print ti, imgFina
		im = Image.open(imgFina)

		reltracker.Add(im, posData[ti])


	reltracker.Train()
	



