#!/usr/bin/python

import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from IPython.display import Image
from keras.optimizers import Adam
import os
import imghdr
import pandas as pd
import tensorflow as tf
import numpy as np

class TrainCNN:
    def __init__(self):
        pass
    
    def manual_cnn_model(self):
        self.cnn = tf.keras.models.Sequential() #Initialising the CNN
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) #Convolution
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #Pooling
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) #Adding a second convolutional layer
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #Second Layer Pooling
        self.cnn.add(tf.keras.layers.Flatten()) #Flattening
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) #Full Connection
        self.cnn.add(tf.keras.layers.Dense(units=3, activation='softmax')) #Output Layer
        return self.cnn
    
    def mobilenet_pretrained_model(self):
        self.base_model=keras.applications.MobileNet(input_shape = (64,64,3), include_top=False, weights='imagenet', classes=1000) #imports the MobileNetV2 model and discards the last 1000 neuron layer.
        self.x = self.base_model.output
        self.x = GlobalAveragePooling2D()(self.x)
        self.x = Dense(128,activation='relu')(self.x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        self.x = Dense(128,activation='relu')(self.x) #dense layer 2
        self.preds = Dense(3,activation='softmax')(self.x) #final layer with softmax activation for 3 classes

        self.mobilenet = Model(inputs=self.base_model.input,outputs=self.preds) #specify the inputs and outputs
        return self.mobilenet
    
    def preprocess_training_set(self):
        self.train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 270)
        self.training_set = self.train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
        return self.training_set
      
    def preprocess_test_set(self):
        self.test_datagen = ImageDataGenerator(rescale = 1./255)
        self.test_set = self.test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
        return self.test_set
    
    def train_model(self, model, training_set, test_set, epochs_n = 5):
        model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.fit(x = training_set, validation_data = test_set, epochs = epochs_n)
        return model
        
    def save_trained_model(self, direc, model_results):
        model_results.save(direc)
    
