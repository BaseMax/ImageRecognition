import os
import shutil
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# tensorflow
# https://keras.io/losses/
##################################################################
dataset = 'corel'
output = 'output'
##################################################################
trains = []
tests = []
##################################################################
# $output/
if not os.path.exists(output):
	os.mkdir(output)
##################################################################
# $output/test
test = os.path.join(output, 'test')
if not os.path.exists(test):
	os.mkdir(test)
##################################################################
# $output/train
train = os.path.join(output, 'train')
if not os.path.exists(train):
	os.mkdir(train)
##################################################################
for index in range(100):
	trains.append(os.path.join(train, str(index)))
	if not os.path.exists(trains[-1]):
		os.mkdir(trains[-1])
	##################################################################
	tests.append(os.path.join(test, str(index)))
	if not os.path.exists(tests[-1]):
		os.mkdir(tests[-1])
	##################################################################
	# 80% : train : 80/100
	# 20% : test : 20/100
	##################################################################
	files = ['{}_{}.jpg'.format(index, (index*100)+prefix) for prefix in range(1,81)] # 1 - 80
	# files = ['{}_{}.jpg'.format(index, (index*100)+prefix) for prefix in range(1,9)] # 1 - 8 # For Testing
	for file in files:
		# https://docs.python.org/3/library/os.path.html?highlight=os%20path%20join#os.path.join
		source = os.path.join(dataset, file)
		# print(source)
		# if os.path.exists(source):
		destination = os.path.join(trains[index], file)
		# print(destination)
		# https://docs.python.org/3/library/shutil.html
		shutil.copyfile(source, destination)
	##################################################################
	files = ['{}_{}.jpg'.format(index, (index*100)+prefix) for prefix in range(81,101)] # 80 - 100
	# files = ['{}_{}.jpg'.format(index, (index*100)+prefix) for prefix in range(9,11)] # 8 - 10 # For Testing
	for file in files:
		source = os.path.join(dataset, file)
		# if os.path.exists(source):
		destination = os.path.join(tests[index], file)
		# https://docs.python.org/3/library/shutil.html#shutil.copyfile
		shutil.copyfile(source, destination)
##################################################################
# https://keras.io/models/sequential/
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(100, activation='softmax'))
# https://keras.io/models/model/#compile
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
##################################################################
# https://keras.io/preprocessing/image/#imagedatagenerator-class
# https://keras.io/preprocessing/image/#imagedatagenerator-methods
# scale 1./255
trainData = ImageDataGenerator(rescale=1./255)
testData = ImageDataGenerator(rescale=1./255)
# https://keras.io/preprocessing/image/#flow_from_directory
# https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
trainEngine = trainData.flow_from_directory(train, class_mode='categorical', target_size=(200, 200), batch_size=20)
##################################################################
for data, label in trainEngine:
	print('Label size:', label.shape)
	print('Data size:', data.shape)
	break
##################################################################
history = model.fit_generator(trainEngine, epochs=30, steps_per_epoch=100)
# history = model.fit_generator(trainEngine, epochs=3, steps_per_epoch=10) ## For testing
##################################################################
acc = history.history['acc']
valAccuracy = history.history['val_acc']
loss = history.history['loss']
valLoss = history.history['val_loss']
##################################################################
epochs = range(len(acc)) ## e.g: 30
##################################################################
plt.plot(epochs, acc, 'bo', label='Training')
plt.plot(epochs, valAccuracy, 'b', label='Validation')
plt.title('The Accuracy of Training and Validation ')
plt.legend()
##################################################################
plt.figure()
##################################################################
plt.plot(epochs, loss, 'bo', label='Loss Training')
plt.plot(epochs, valLoss, 'b', label='Loss Validation')
plt.title('The Loss of Training and Validation')
plt.legend()
##################################################################
plt.show()
##################################################################
