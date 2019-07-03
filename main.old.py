import glob
import math
import numpy
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import os

os.chdir("corel/")
images=glob.glob("*.jpg")
imagesGroup=[]
# print(images)

for imageFile in images:
	names=imageFile.split("_")
	# print(names)
	# print(names)
	# names[0] ==> group name
	# imagesGroup[names[0]] ... (imageFile)
	# continue
	image=cv2.imread(imageFile, 0) # 0 means: not BGR!
	cv2.imshow('title', image)
	print(image)
	width=image.shape[1]
	height=image.shape[0]
	average=numpy.zeros((height, width))
	variance=numpy.zeros((height, width))
	stdVariance=numpy.zeros((height, width))
	for y in range(height):
		for x in range(width):
			sum=0
			count=0
			avg=0
			for yi in range(y-1, y+1+1):
				for xi in range(x-1 ,x+1+1):
					if yi >= 0 and yi < height and xi >= 0 and xi < width:
						sum+=image[yi][xi]
						count+=1
			avg=sum / count
			sum2=0
			for yi in range(y-1, y+1+1):
				for xi in range(x-1 ,x+1+1):
					try:
						sum2+=(image[yi][xi] - avg) ** 2
					except IndexError:
						sum2+=0
			var=sum2 / count
			print( x,",", y, "==>", sum, sum2, count, avg, var)
			variance[y][x]=var
			average[y][x]=avg
	for y in range(height):
		for x in range(width):
			stdVariance[y][x]=math.sqrt(variance[y][x])
	print(variance)
	print(average)
	print(stdVariance)

# print(imagesGroup)
