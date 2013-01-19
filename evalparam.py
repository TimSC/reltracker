import sys, os
from PIL import Image
from reltracker import ReadPosData, RelTracker

def EvalParam(settings):
	assert os.path.exists(sys.argv[1])
	posData = ReadPosData(sys.argv[1])

	assert os.path.exists(sys.argv[2])
	testData = ReadPosData(sys.argv[2])

	reltracker = RelTracker()
	reltracker.settings = settings

	#Add training data to tracker
	for ti in posData:
		imgFina = sys.argv[3]+"/{0:05d}.png".format(ti)
		assert os.path.exists(imgFina)
		print ti, imgFina
			
		im = Image.open(imgFina)

		reltracker.Add(im, posData[ti])

	#Train the tracker
	reltracker.Train()
	reltracker.ClearTraining() #Remove data that cannot be pickled

	if 1:
		out = {}
		frameNum = 9000
		currentPos = None
		while 1:
			#Load current frame
			imgFina = sys.argv[3]+"/{0:05d}.png".format(frameNum)
			if not os.path.exists(imgFina):
				break
			print "frameNum", frameNum
			im = Image.open(imgFina)

			if currentPos is not None:
				#Predict position on current frame
				try:
					currentPos = reltracker.Predict(im, currentPos)
				except Exception as err:
					print err
					currentPos = None

			if frameNum in testData:
				truePos = testData[frameNum]
				if currentPos is not None:
					#Check against known positions		
					err = []
					for p, t in zip(currentPos, truePos):
						diff = abs(((p[0] - t[0])**2.+(p[1] - t[1])**2.)**0.5)
						err.append(diff)
					out[frameNum] = err

				currentPos = truePos

			#Go to next frame
			frameNum += 1
	return out
			
if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "Usage:",sys.argv[0],"markedPositions.dat testPositions.dat /path/to/images"
		exit(0)
	
	outFi = open("evalparam2posrot.txt","wt")
	for i in [10,20,40,80,160,300,500]:
		settings = [{'shapeNoise':12, 'cloudEnabled':1, 'supportMaxOffset':39, 'trainVarianceOffset': 41,\
					'rotationVar': 0.1, 'numTrainingOffsets':5000, 'numSupportPix':i}, \
					{'shapeNoise':100, 'cloudEnabled':0, 'supportMaxOffset':20, 'trainVarianceOffset': 5,\
					'rotationVar': 0.1, 'numTrainingOffsets':5000, 'numSupportPix':i}]

		errs = EvalParam(settings)
		outFi.write("Settings:"+str(settings)+"\n")
		outFi.write("Errors:"+str(errs)+"\n")
		outFi.flush()


