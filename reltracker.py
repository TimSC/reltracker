
from PIL import Image
import time, math, pickle, sys, os, copy
import numpy as np
import sklearn.ensemble as ensemble
from relutil import GetPixIntensityAtLoc, ToGrey

#******* Utility functions

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
	"""
	RelAxis represents a regression model in a particular axis. A tracking
	point generally requires (at least) two RelAxis objects for 2D tracking.
	"""

	def __init__(self):
		self.reg = None
		self.trainingData = []
		self.verbose = 1
		self.shapeNoise = 12
		self.cloudEnabled = 1
		self.supportPixOffset = None

	def Add(self, im, pos):
		"""
		Add an annotated frame with a set of tracker positions for later 
		training of the regression model.
		"""
		self.trainingData.append((im, pos))

	def ClearTraining(self):
		"""
		Clear training data from this object. This should allow the object
		to be pickled.
		"""
		self.trainingData = []

	def Train(self):
		"""
		Train a regression model based on the added training data. This class only
		considers a single axis for a single tracking point.
		"""
		assert self.supportPixOffset is not None

		#Convert to grey scale, numpy array
		greyPix = np.empty((len(self.trainInt), len(self.trainInt[0])))
		for rowNum, trainIntensity in enumerate(self.trainInt):
			for pixNum, col in enumerate(trainIntensity):
				greyPix[rowNum, pixNum] = ToGrey(col)

		#Calculate relative position of other points in cloud
		#Note: this implementation is not efficiant as the distances are
		#repeatedly recalculated!
		if self.cloudEnabled: 
			if self.verbose: 
				print "Calc distances"
				sys.stdout.flush()
			trainCloudPos = []
			for frameNum, offset, rotation in \
				zip(self.trainFra, self.trainOff, self.trainRot):
				cloudPosOnFrame = []
				posOnFrame = self.trainingData[frameNum][1]
				for trNum, pos in enumerate(posOnFrame):
					if trNum == self.trackerNum:
						continue #Skip distance to self

					#Rotate training offset vector
					offsetRX = math.cos(rotation) * offset[0] - math.sin(rotation) * offset[1]
					offsetRY = math.sin(rotation) * offset[0] + math.cos(rotation) * offset[1]

					#Calculate unrotated diff vector to cloud position
					diffX = pos[0] - (posOnFrame[self.trackerNum][0])
					diffY = pos[1] - (posOnFrame[self.trackerNum][1])

					#Rotate the cloud position vector
					diffRX = math.cos(rotation) * diffX - math.sin(rotation) * diffY
					diffRY = math.sin(rotation) * diffX + math.cos(rotation) * diffY

					#Modify cloud position with synthetic training offset
					offsetDiffRX = diffRX - offsetRX
					offsetDiffRY = diffRY - offsetRY

					if self.axis == "x":
						cloudPosOnFrame.append(offsetDiffRX)
					else:
						cloudPosOnFrame.append(offsetDiffRY)
				trainCloudPos.append(cloudPosOnFrame)

			#Add noise to cloud positions
			trainCloudPos = np.array(trainCloudPos)
			trainCloudPos = trainCloudPos + np.random.randn(*trainCloudPos.shape) * self.shapeNoise

		#Select axis labels
		trainOffArr = np.array(self.trainOff)
		if self.axis == "x":
			labels = trainOffArr[:,0]
		else:
			labels = trainOffArr[:,1]
	
		#If selected, merge the cloud position data with pixel intensities
		if self.cloudEnabled:
			trainDataFinal = np.hstack((greyPix, trainCloudPos))
		else:
			trainDataFinal = greyPix

		#Train regression model
		self.reg = ensemble.GradientBoostingRegressor()
		self.reg.fit(trainDataFinal, labels)

	def Predict(self, im, pos):
		"""
		Request a prediction based on a specified image and tracker position arrangement. The
		resulting prediction is for a single axis and a single point.
		"""

		assert self.reg is not None #A regression model must first be trained
		currentPos = copy.deepcopy(pos)
		pix = GetPixIntensityAtLoc(np.array(im), im, self.supportPixOffset, 
			currentPos[self.trackerNum][0], currentPos[self.trackerNum][1])
		if pix is None:
			raise Exception("Pixel intensities could not be determined")

		#Convert to grey scale
		greyPix = []
		for col in pix:
			greyPix.append(ToGrey(col))

		#Calculate relative distances to cloud
		cloudPosOnFrame = []
		for trNum, trPos in enumerate(pos):
			if trNum == self.trackerNum:
				continue #Skip distance to self
			xdiff = trPos[0] - pos[self.trackerNum][0]
			ydiff = trPos[1] - pos[self.trackerNum][1]

			if self.axis == "x":
				cloudPosOnFrame.append(xdiff)
			else:
				cloudPosOnFrame.append(ydiff)

		if self.cloudEnabled:
			testData = np.concatenate((greyPix, cloudPosOnFrame))
		else:
			testData = greyPix

		#Make prediction
		pred = self.reg.predict(testData)[0]

		if self.axis == 'x': axisNum = 0
		else: axisNum = 1

		#Adjust position
		if not isinstance(currentPos[self.trackerNum], list): #Ensure this is a list
			currentPos[self.trackerNum] = list(currentPos[self.trackerNum])
		currentPos[self.trackerNum][axisNum] -= pred

		return currentPos
	
	def TrainingComplete(self):
		if self.reg is not None: return True
		return False

#****************************************************

class RelTracker:
	"""
	An implementation of "Non-linear Predictors for Facial feature Tracking 
	across Pose and Expression". This class contains multiple regression
	models for prediction tracking on video frames. 

	The name reltracker comes from RELative Tracker, because the relative
	positions of the other tracking points are included in the prediction
	model.

	1) Add some annotated frames using Add()
	2) Train a model by Train()
	3) Predict new positions by using Predict()

	If used for scientific purposes, please cite:

	Tim Sheerman-Chase, Eng-Jon Ong, Richard Bowden. Non-linear Predictors 
	for Facial feature Tracking across Pose and Expression. In IEEE 
	Conference on Automatic Face and Gesture Recognition, Shanghai, 2013.

	"""

	def __init__(self):
		self.trainingData = []

		self.trainingIntLayers = []
		self.trainingOffLayers = []
		self.trainingRotLayers = []
		self.trainingFraLayers = []

		self.numIterations = 5
		self.scalePredictors = None
		self.serialTraining = None
		self.supportPixOffset = []
		self.numSupportPix = [200, 200] #[500, 500]
		self.maxSupportOffset = [39, 20]
		self.trainingIntLayers = None
		self.trainVarianceOffset = [41, 5]
		self.rotationVar = [0., 0.]
		self.numTrainingOffsets = [2000, 2000] #[5000, 5000]
		self.settings = [{'shapeNoise':12, 'cloudEnabled':1, 'trainVarianceOffset': 41},
				{'shapeNoise':100, 'cloudEnabled':0}]

	def Add(self, im, pos):
		"""
		Add an annotated frame with a set of tracker positions for later 
		training of the regression model.
		"""
		if im.mode != "RGB" and im.mode != "L": 
			im = im.convert("RGB")

		self.trainingData.append((im, pos))
		assert(len(self.trainingData[0][1]) == len(self.trainingData[-1][1]))

	def ClearTraining(self):
		"""
		Clear training data from this object. This should allow the object
		to be pickled.
		"""
		self.trainingData = []
		for layerNum, layer in enumerate(self.scalePredictors):
			for relaxis in layer:
				relaxis.ClearTraining()

	def Train(self):
		"""
		Train regression models based on the added training data. This class 
		considers multiple tracking points in 2D tracking (as in 2 axis are
		used for each tracking point.)
		"""

		while self.GetProgress() < 1.:
			self.ProgressTraining()

	def GenerateTrainingIntensities(self, layerNum, trNum, supportPixOffset):

		layerTrainVarOffset = self.trainVarianceOffset[layerNum]
		layerRotationVar = self.rotationVar[layerNum]
		layerNumTrainingOffsets = self.numTrainingOffsets[layerNum]
		outTrainInt, outTrainOffsets, outTrainRot, outTrainFra = [], [], [], []

		for frameNum, (im, pos) in enumerate(self.trainingData):
			trPos = pos[trNum]

			trainRotation = np.random.randn() * layerRotationVar
			trainOffset = np.random.randn(2) * layerTrainVarOffset

			offset = (trainOffset[0] + trPos[0], trainOffset[1] + trPos[1])

			pix = GetPixIntensityAtLoc(np.array(im), im, supportPixOffset, offset[0], offset[1], trainRotation)
			if pix is None:
				#Pixel is outside of image: discard this training offset
				continue

			outTrainInt.append(pix)
			outTrainOffsets.append(trainOffset)
			outTrainRot.append(trainRotation)
			outTrainFra.append(frameNum)

		return outTrainInt, outTrainOffsets, outTrainRot, outTrainFra


	def ProgressTraining(self):

		assert(len(self.trainingData)>0)
		numTrackers = len(self.trainingData[0][1])

		if self.scalePredictors is None:
			self.scalePredictors = []
			self.supportPixOffset = []

			#For each layer of hierarchy
			for layerSettings, layerNumSupportPix, layerMaxSupportOffset \
				in zip(self.settings, self.numSupportPix, self.maxSupportOffset):
				layer = []
				layerPixOffset = []

				#For each tracker
				for trNum in range(numTrackers):

					trSupportPixOffset = np.random.uniform(-layerMaxSupportOffset, 
						layerMaxSupportOffset, (layerNumSupportPix, 2))
					layerPixOffset.append(trSupportPixOffset)

					#Create two axis trackers
					for axis in ['x', 'y']:
						relaxis = RelAxis()
						relaxis.trackerNum = trNum
						relaxis.axis = axis
						relaxis.supportPixOffset = trSupportPixOffset
						for settingKey in layerSettings:
							setattr(relaxis, settingKey, layerSettings[settingKey])
						for td in self.trainingData:
							relaxis.Add(*td)
						layer.append(relaxis)

				self.scalePredictors.append(layer)
				self.supportPixOffset.append(layerPixOffset)
			return
	
		#Generate support pixel intensity container structure
		if self.trainingIntLayers is None:
			self.trainingIntLayers = []
			self.trainingOffLayers = []
			self.trainingRotLayers = []
			self.trainingFraLayers = []

			for layer in self.scalePredictors:
				self.trainingIntLayers.append([])
				self.trainingOffLayers.append([])
				self.trainingRotLayers.append([])
				self.trainingFraLayers.append([])

				for relaxis in layer:
					self.trainingIntLayers[-1].append([])
					self.trainingOffLayers[-1].append([])
					self.trainingRotLayers[-1].append([])
					self.trainingFraLayers[-1].append([])


		for layerNum, (layer, layerSupportPixOffset, trainingIntLayer, trainingIntsL, trainingOffL, trainingRotL, trainFraL) in \
			enumerate(zip(self.scalePredictors, self.supportPixOffset, self.trainingIntLayers, 
				self.trainingIntLayers, self.trainingOffLayers, self.trainingRotLayers, self.trainingFraLayers)):
			for trNum, (supportPixOffset, ints, offs, rots, fra) in \
				enumerate(zip(layerSupportPixOffset, trainingIntsL, trainingOffL, trainingRotL, trainFraL)):
				if len(ints) >= self.numTrainingOffsets[layerNum]:
					continue #Skip completed tracker's intensities

				trainInt, trainOffsets, trainRot, trainFra = self.GenerateTrainingIntensities(layerNum, trNum, supportPixOffset)
				
				ints.extend(trainInt)
				offs.extend(trainOffsets)
				rots.extend(trainRot)
				fra.extend(trainFra)
				print len(ints)
				return
	
		#Train individual axis predictors
		for layerNum, layer in enumerate(self.scalePredictors):
			for relaxis in layer:
				if relaxis.TrainingComplete(): continue #Skip completed trackers

				print "Training", layerNum, relaxis.trackerNum, relaxis.axis
				sys.stdout.flush()
				relaxis.trainInt = self.trainingIntLayers[layerNum][relaxis.trackerNum]
				relaxis.trainOff = self.trainingOffLayers[layerNum][relaxis.trackerNum]
				relaxis.trainRot = self.trainingRotLayers[layerNum][relaxis.trackerNum]
				relaxis.trainFra = self.trainingFraLayers[layerNum][relaxis.trackerNum]

				relaxis.Train()
				return
				
	def Predict(self, im, pos):
		"""
		Request predictions based on a specified image and tracker position arrangement. The
		resulting prediction is for all tracking points.
		"""

		assert self.scalePredictors is not None #Train the model first!
		assert len(pos) == len(self.scalePredictors[0]) / 2
		if im.mode != "RGB" and im.mode != "L": 
			im = im.convert("RGB")
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

	def PrepareForPickle(self):
		assert self.serialTraining is None
		self.serialTraining = []
		for im, pos in self.trainingData:
			self.serialTraining.append((dict(data=im.tostring(), size=im.size, mode=im.mode), pos))
		self.ClearTraining()

	def PostUnPickle(self):
		assert self.serialTraining is not None
		self.trainingData = []
		for imDat, pos in self.serialTraining:
			im = Image.fromstring(**imDat)
			self.trainingData.append((im, pos))

		#Set training data in axis objects
		for layerNum, layer in enumerate(self.scalePredictors):
			for relaxis in layer:
				relaxis.ClearTraining()
				for tr in self.trainingData:
					relaxis.Add(*tr)

		self.serialTraining = None

	def GetProgress(self):

		if self.scalePredictors is None:
			return 0.
		countDone = 1 #Counting the scale predictor initialisation as a valid step
		countTotal = 1
		for layerNum, layer in enumerate(self.scalePredictors):
			for relaxis in layer:
				countDone += relaxis.TrainingComplete()
				countTotal += 1
		return float(countDone) / countTotal
		
	def Update(self):
		if self.GetProgress() >= 1.:
			return
		self.ProgressTraining()

#************************************************************

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "Usage:",sys.argv[0],"markedPositions.dat /path/to/images"
		exit(0)

	assert os.path.exists(sys.argv[1])
	posData = ReadPosData(sys.argv[1])

	if 1:
		reltracker = RelTracker()

		#Add training data to tracker
		for ti in posData:
			imgFina = sys.argv[2]+"/{0:05d}.png".format(ti)
			assert os.path.exists(imgFina)
			print ti, imgFina
			
			im = Image.open(imgFina)

			reltracker.Add(im, posData[ti])

		#Train the tracker
		reltracker.Train()
		reltracker.ClearTraining() #Remove data that cannot be pickled

		#reltracker.PrepareForPickle()
		#print "Pickling"
		#pickle.dump(reltracker, open("tracker.dat","wb"), protocol = -1)

	if 1:
		#print "Unpickling"
		#reltracker = pickle.load(open("tracker.dat","rb"))
		#reltracker.PostUnPickle()

		frameNum = 0
		currentPos = None
		while frameNum < 400:
			#Load current frame
			imgFina = sys.argv[2]+"/{0:05d}.png".format(frameNum)
			if not os.path.exists(imgFina):
				break
			print "frameNum", frameNum
			im = Image.open(imgFina)

			if frameNum in posData:
				#Use existing known positions
				currentPos = posData[frameNum]
			elif currentPos is not None:
				#Predict position on current frame
				try:
					currentPos = reltracker.Predict(im, currentPos)
				except Exception as err:
					print err
					currentPos = None
			
			#Visualise tracking
			iml = im.load()
			if currentPos is not None: 
				for pos in currentPos:
					for i in [-1,0,+1]:
						for j in [-1,0,+1]:
							col = (255,255,255)
							if len(im.mode)==1: col = 255
							try:
								iml[int(round(pos[0]+i)),int(round(pos[1]+j))] = col
							except:
								pass
			im.save("{0:05d}.jpg".format(frameNum))
	
			#Go to next frame
			frameNum += 1

			


