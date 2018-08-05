#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:51:02 2018

@author: justindodson
"""

# Creating a simple image recognition pipeline 

# PART 1 - Building the CNN

# import the required packages and libraries 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN 
classifier = Sequential()

# Step 1 - Convolution Layer
classifier.add(Convolution2D(filters = 32, 
                             kernel_size = 3, 
                             input_shape = (64, 64, 3),
                             activation = 'relu'))
# Step 2 - Max Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Add second convolutional layer after only getting a 75% on the first model
classifier.add(Convolution2D(filters = 32, 
                             kernel_size = 3,                              
                             activation = 'relu'))
# add Max Pooling to the second convolution layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection: Build ANN for Image Classification
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN (apply Gradient Descent)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# PART 2 - FITTING THE CNN TO THE IMAGES
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch = 8000,
                        epochs = 25,
                        validation_data = test_set,
                        validation_steps = 2000)



# PART 3 - MAKING A NEW PREDICTION
import numpy as np
from keras.preprocessing import image

test_image_1 = image.load_img('dataset/predictions/dog_or_cat_1.jpg', target_size = (64, 64))
test_image_2 = image.load_img('dataset/predictions/dog_or_cat_2.jpg', target_size = (64, 64))
test_image_3 = image.load_img('dataset/predictions/dog_or_cat_3.JPG', target_size = (64, 64))
test_image_4 = image.load_img('dataset/predictions/dog_or_cat_4.JPG', target_size = (64, 64))
test_image_5 = image.load_img('dataset/predictions/dog_or_cat_5.JPG', target_size = (64, 64))
test_image_6 = image.load_img('dataset/predictions/dog_or_cat_6.JPG', target_size = (64, 64))
test_image_7 = image.load_img('dataset/predictions/dog_or_cat_7.JPG', target_size = (64, 64))

test_image_1 = image.img_to_array(test_image_1)
test_image_2 = image.img_to_array(test_image_2)
test_image_3 = image.img_to_array(test_image_3)
test_image_4 = image.img_to_array(test_image_4)
test_image_5 = image.img_to_array(test_image_5)
test_image_6 = image.img_to_array(test_image_6)
test_image_7 = image.img_to_array(test_image_7)

test_image_1 = np.expand_dims(test_image_1, axis = 0)
test_image_2 = np.expand_dims(test_image_2, axis = 0)
test_image_3 = np.expand_dims(test_image_3, axis = 0)
test_image_4 = np.expand_dims(test_image_4, axis = 0)
test_image_5 = np.expand_dims(test_image_5, axis = 0)
test_image_6 = np.expand_dims(test_image_6, axis = 0)
test_image_7 = np.expand_dims(test_image_7, axis = 0)

result1 = classifier.predict(test_image_1)
result2 = classifier.predict(test_image_2)
result3 = classifier.predict(test_image_3)
result4 = classifier.predict(test_image_4)
result5 = classifier.predict(test_image_5)
result6 = classifier.predict(test_image_6)
result7 = classifier.predict(test_image_7)

training_set.class_indices

if result1[0][0] == 1:
    prediction1 = 'dog'
else:
    prediction1 = 'cat'
    
if result2[0][0] == 1:
    prediction2 = 'dog'
else:
    prediction2 = 'cat'
    
if result3[0][0] == 1:
    prediction3 = 'dog'
else:
    prediction3 = 'cat'
    
if result4[0][0] == 1:
    prediction4 = 'dog'
else:
    prediction4 = 'cat'
    
if result5[0][0] == 1:
    prediction5 = 'dog'
else:
    prediction5 = 'cat'
    
if result6[0][0] == 1:
    prediction6 = 'dog'
else:
    prediction6 = 'cat'
    
if result7[0][0] == 1:
    prediction7 = 'dog'
else:
    prediction7 = 'cat'


print("Prediction for Image: dog_or_cat_1.jpg (DOG) = " + prediction1)
print("Prediction for Image: dog_or_cat_2.jpg (DOG) = " + prediction2)
print("Prediction for Image: dog_or_cat_3.jpg (CAT) = " + prediction3)
print("Prediction for Image: dog_or_cat_4.jpg (DOG) = " + prediction4)
print("Prediction for Image: dog_or_cat_5.jpg (CAT) = " + prediction5)
print("Prediction for Image: dog_or_cat_6.jpg (DOG) = " + prediction6)
print("Prediction for Image: dog_or_cat_7.jpg (DOG) = " + prediction7)




"""
After training the modl and sending in data images to predict, I found that the CNN was able to 
accuratly classify 6/7 images correctly. One image of a cat was mislabled as a dog in image 3.

The accuracy was at 80% after second run of training but overall decent. 
This model could use some more parameter tuning to make it more accurate network.
"""


















