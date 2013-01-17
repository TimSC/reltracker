
from PIL import Image
import time, math, pickle, sys, os, copy
import numpy as np
import sklearn.ensemble as ensemble

#******* Utility functions

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

def GetPixIntensityAtLoc(iml, supportOffsets, loc, rotation = 0.):
	out = []
	for offset in supportOffsets:
		#Apply rotation (anti-clockwise)
		rx = math.cos(rotation) * offset[0] - math.sin(rotation) * offset[1]
		ry = math.sin(rotation) * offset[0] + math.cos(rotation) * offset[1]

		#Get pixel at this location
		try:
			out.append(BilinearSample(iml, rx + loc[0], ry + loc[1]))
		except IndexError:
			return None
	return out

#*******************************************************************************

class RelAxis:
	def __init__(self):
		self.numSupportPix = 500
		self.numTrainingOffsets = 5000
		self.maxSupportOffset = 30
		self.reg = None
		self.trainingData = None

	def Train(self):

		#Generate support pix and training offsets
		self.supportPixOffset = np.random.uniform(-self.maxSupportOffset, 
				self.maxSupportOffset, (self.numSupportPix, 2))

		#Create pixel access objects
		trainImgPix = [train[0].load() for train in self.trainingData]

		#Get pixel intensities at training offsets
		trainPix = []
		trainOffsetsX = []
		trainOffsetsY = []
		for im, pos in self.trainingData:
			trPos = pos[self.trackerNum]
			iml = im.load()

			for train in range(self.numTrainingOffsets/len(self.trainingData)):
				trainOffset = np.random.randn(2) * self.trainVarianceOffset

				offset = (trainOffset[0] + trPos[0], trainOffset[1] + trPos[1])

				pix = GetPixIntensityAtLoc(iml, self.supportPixOffset, offset)
				if pix is None:
					#Pixel is outside of image: discard this training offset
					continue
				trainPix.append(pix)
				trainOffsetsX.append(trainOffset[0])
				trainOffsetsY.append(trainOffset[1])
			print len(trainPix)
		numValidTraining = len(trainPix)
		assert numValidTraining > 0

		#Convert to grey scale, numpy array
		greyPix = np.empty((numValidTraining, self.numSupportPix))
		for rowNum, trainIntensity in enumerate(trainPix):
			for pixNum, col in enumerate(trainIntensity):
				greyPix[rowNum, pixNum] = ITUR6012(col)

		#Select axis labels
		if self.axis == "x":
			labels = trainOffsetsX
		else:
			labels = trainOffsetsY

		#Train regression model
		self.reg = ensemble.GradientBoostingRegressor()
		self.reg.fit(greyPix, labels)

	def Predict(self, im, pos):
		currentPos = copy.deepcopy(pos)
		pix = GetPixIntensityAtLoc(im.load(), self.supportPixOffset, currentPos[self.trackerNum])
		if pix is None:
			raise Exception("Pixel intensities could not be determined")

		#Convert to grey scale
		greyPix = []
		for col in pix:
			greyPix.append(ITUR6012(col))

		#Make prediction
		pred = self.reg.predict(greyPix)[0]

		if self.axis == 'x': axisNum = 0
		else: axisNum = 1

		#Adjust position
		currentPos[self.trackerNum][axisNum] -= pred

		return currentPos

#****************************************************

class RelTracker:
	def __init__(self):
		self.trainingData = []
		self.numIterations = 5

	def Add(self, im, pos):
		self.trainingData.append((im, pos))
		assert(len(self.trainingData[0][1]) == len(self.trainingData[-1][1]))


	def Train(self):
		
		assert(len(self.trainingData)>0)
		numTrackers = len(self.trainingData[0][1])
		self.scalePredictors = []

		#First layer of hierarchy
		layer = []
		for trNum in range(numTrackers):
			for axis in ['x', 'y']:
				relaxis = RelAxis()
				relaxis.trackerNum = trNum
				relaxis.axis = axis
				relaxis.shapeNoise = 12
				relaxis.cloudEnabled = 1
				relaxis.supportMaxOffset = 39
				relaxis.trainVarianceOffset = 41
				relaxis.rotationVar = 0.1
				relaxis.trainingData = self.trainingData
				layer.append(relaxis)
		self.scalePredictors.append(layer)

		#Second layer of hierarchy
		layer = []
		for trNum in range(numTrackers):
			for axis in ['x', 'y']:
				relaxis = RelAxis()
				relaxis.trackerNum = trNum
				relaxis.axis = axis
				relaxis.shapeNoise = 1
				relaxis.cloudEnabled = 0
				relaxis.supportMaxOffset = 20
				relaxis.trainVarianceOffset = 5
				relaxis.rotationVar = 0.1
				relaxis.trainingData = self.trainingData
				layer.append(relaxis)
		self.scalePredictors.append(layer)
		
		#Train individual axis predictors
		for layerNum, layer in enumerate(self.scalePredictors):
			for relaxis in layer:
				print "Training", layerNum, relaxis.trackerNum, relaxis.axis
				relaxis.Train()
				relaxis.trainingData = None #Remove data that cannot be pickled

	def Predict(self, im, pos):
		assert len(pos) == len(self.scalePredictors[0]) / 2
		currentPos = copy.deepcopy(pos)
		
		#For each layer in the hierarchy,
		for layerNum, layer in enumerate(self.scalePredictors): 

			#For a specified number of iterations,
			for iterNum in range(self.numIterations):

				#For each axis predictor,
				for axisNum, relaxis in enumerate(layer):
					#print "Predict", layerNum, relaxis.trackerNum, relaxis.axis

					#Make a prediction
					currentPos = relaxis.Predict(im, currentPos)

		return currentPos
		
#************************************************************

if __name__ == "__main__":
	posData = ReadPosData(sys.argv[1])

	if 0:
		reltracker = RelTracker()
		for ti in posData:
			imgFina = sys.argv[2]+"/{0:05d}.png".format(ti)
			print ti, imgFina
			im = Image.open(imgFina)

			reltracker.Add(im, posData[ti])


		reltracker.Train()
		reltracker.trainingData = None #Remove data that cannot be pickled

		pickle.dump(reltracker, open("tracker.dat","wb"), protocol = -1)

	if 1:
		reltracker = pickle.load(open("tracker.dat","rb"))

		frameNum = 0
		currentPos = None
		while 1:
			imgFina = sys.argv[2]+"/{0:05d}.png".format(frameNum)
			if not os.path.exists(imgFina):
				break
			print "frameNum", frameNum
			im = Image.open(imgFina)
			if im.mode != "RGB": im = im.convert("RGB")

			if frameNum in posData:
				currentPos = posData[frameNum]
			elif currentPos is not None:
				#Predict position on current frame
				try:
					currentPos = reltracker.Predict(im, currentPos)
				except:
					currentPos = None
			
			#Visualise tracking
			iml = im.load()
			if currentPos is not None: 
				for pos in currentPos:
					for i in [-1,0,+1]:
						for j in [-1,0,+1]:
							iml[pos[0]+i,pos[1]+j] = (255,255,255)
			im.save("{0:05d}.jpg".format(frameNum))
	
			#Go to next frame
			frameNum += 1

			


