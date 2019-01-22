import numpy as np
import tensorflow as tf
import csv
import os
import matplotlib.pyplot as plt
from generator import generator
import cv2
from scipy import ndimage

from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D
from sklearn.model_selection import train_test_split


correction = 0.2
center_only = 0
#mirror = 0  ## Used in model0.h5
mirror  = 1
#batch_size = 32 ## Used in model0.h5
#batch_size = 16 ## Used in model1.h5
batch_size = 4
#batch_size = 256

crop_up = 50
crop_low = 20
ch = 3
row = 160 - crop_up - crop_low
col  = 320

"""
## Using without generator
 
car_images = []
car_angles = []

with open('../../../opt/carnd_p3/data/driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if (i == 0):
                i = i+1
            elif (i < 500):           
                steering_center = float(row[3])           
                img_center = np.asarray(ndimage.imread('../../../opt/carnd_p3/data/' + row[0]))
                img_center_flip = np.fliplr(img_center)
                steering_center_flip = - steering_center
                car_images.append(img_center)
                car_images.append(img_center_flip)
                car_angles.append(steering_center)
                car_angles.append(steering_center_flip)
                i = i +1
               
car_images = np.asarray(car_images)
car_angles = np.asarray(car_angles)
"""


###Using generator
samples = []

## Read in driving log files
for i in range(30): 
    # i in range(26) for model0.h5
    # path in model0.h5
    #path = '../../../opt/carnd_p3_own/data_own/driving_log_' + str(i) +'.csv'
    path = '../../../opt/carnd_p3_own/data_own_mountain/driving_log_' + str(i) +'.csv'
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

## Split train and test data

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size, correction, center_only, mirror)
validation_generator = generator(validation_samples, batch_size,correction, center_only, mirror)


## Define model structure
model = Sequential()
# Normalize image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# Crop image
model.add(Cropping2D(cropping=((crop_up,crop_low), (0,0)), input_shape=(160,320, 3)))
# Conv and following layers

#NVIDIA Structure
model.add(Conv2D(24, kernel_size=(5, 5), strides = (2,2), activation='relu', input_shape=(row, col, ch)))
model.add(Conv2D(36, kernel_size = (5,5), strides =(2,2), activation ='relu'))
model.add(Conv2D(48, kernel_size =(5,5), strides = (2,2), activation ='relu'))
model.add(Conv2D(64, kernel_size  =(3,3), activation = 'relu'))
model.add(Conv2D(64, kernel_size= (3,3), activation = 'relu'))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

## My Structure
""" 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(row, col, ch)))
model.add(Conv2D(64, (2,2), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
"""

## Training and validation

model.compile(loss='mse', optimizer='adam')
#model = load_model('model0.h5')

## Using generator, nb_epoch = 3 is used in model0.h5 and model1.h5
history_object = model.fit_generator(train_generator, steps_per_epoch =
            len(train_samples), validation_data=validation_generator,
            validation_steps=len(validation_samples), nb_epoch=5, verbose = 1)


## Without using generator
#model.fit(car_images, car_angles, epochs=1, validation_split=0.2,verbose = 1)
#history_object = model.fit(car_images, car_angles, epochs=2, validation_split=0.2, verbose = 1)

## Save model
k = 2 
model_name = 'model'+ str(k) + '.h5'
model.save(model_name)
model_weights = 'model_' + str(k) +'_weigths.h5'

model.save_weights(model_weights)
## Visualization 
### print the keys contained in the history object

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()



