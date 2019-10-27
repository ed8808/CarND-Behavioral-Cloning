import os
import csv

samples = []
with open('./data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    samples = samples[1:]
    
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import math
import random

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
          
            images = []
            angles = []
            for batch_sample in batch_samples:
                _img_center = cv2.imread('./data2/IMG/'+batch_sample[0].split('/')[-1])
                _img_left = cv2.imread('./data2/IMG/'+batch_sample[1].split('/')[-1])
                _img_right = cv2.imread('./data2/IMG/'+batch_sample[2].split('/')[-1])
                
                img_center = cv2.cvtColor(_img_center, cv2.COLOR_BGR2RGB)
                img_left = cv2.cvtColor(_img_left, cv2.COLOR_BGR2RGB)
                img_right = cv2.cvtColor(_img_right, cv2.COLOR_BGR2RGB)                
                
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                images.append(img_center)
                angles.append(steering_center)
                images.append(np.fliplr(img_center))
                angles.append(-steering_center)
                
                images.append(img_left)
                angles.append(steering_left)
                images.append(np.fliplr(img_left))
                angles.append(-steering_left)
                
                images.append(img_right)
                angles.append(steering_right)
                images.append(np.fliplr(img_right))
                angles.append(-steering_right)
                
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, 
                 input_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

# LeNet CNN
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''
# NVIDIA CNN
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
'''
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=math.ceil(len(validation_samples)/batch_size),
                    epochs=5, verbose=1)

model.save('model.h5')
'''
### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
