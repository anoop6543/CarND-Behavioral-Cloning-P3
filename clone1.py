## UDACITY CarND Project 3 - Behavioral Cloning

##Importing Libraries

import os
import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

#import lines(excel rows data) from driving_log CSV 

lines = []

with open('./trainingdata1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#Declaring 
#Images Array 
#Measurements Array 
#c-factor - Correction factor to include measurements of Left & Right Images with respect to center images

images = []
measurements = []
c_factor = 0.2

#Path to read center, left and right images in the local directory
#Eliminate the unnecessary path and add the filename to local path in the remote server
#Append the read images and the corresponding measurements to the respective arrays


for line in lines:	
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = "./trainingdata1/IMG/" + filename
		image = cv2.imread(local_path)
		images.append(image)
	
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement+c_factor)
	measurements.append(measurement-c_factor)

"""  
***** Code segment for Augementing images by flipping data *****

aug_images = []
aug_measurements = []

for image, measuresurement in zip(images, measurements):
	aug_images.append(image)
	aug_measurements.append(measurement)
	flipped_image = np.fliplr(image)
	flipped_measurement = -measurement
	aug_images.append(flipped_image)
	aug_measurements.append(flipped_measurement)

**** End Code segment for Augmenting data ****
"""
print("image shape is : " + str(image))
print("current_path is :" + str(local_path))

print("Number of Images: " + str(len(images)))
print("Number of Measurements: " + str(len(measurements)))

X_train = np.array(images)
y_train = np.array(measurements) 

print("Converted!")
print("Training Data Shape: " + str(X_train.shape))

# *** Training Model Used is similar to NVIDIA *** 
model = Sequential()

#Normalizing data set
model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,3)))

#Cropping Images
model.add(Cropping2D(cropping=((50,20),(0,0))))

#Beging Convolution Layers Folowed by Maxpooling
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

# Applying Flattening, Dense & Drop Layer
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

#Using Adam optimizer with a learning rate of 1e-4, Loss Function is MSE
model.compile(optimizer=Adam(lr=(1e-4)), loss='mse')

#Using 25 percent of training data for validation set - considered as per suggested in the tutorial
model.fit(X_train, y_train, validation_split=0.25,shuffle=True, nb_epoch=10)

model.save('model_correct.h5')

"""  End Program """
