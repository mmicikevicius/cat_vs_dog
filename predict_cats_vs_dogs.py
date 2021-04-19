#!/usr/bin/python

# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
import imghdr
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import os
import pandas as pd
from sys import argv

# Pre-trained manually created model:
#load and test on new images
def cats_vs_dogs(model, directory):
    new_model = tf.keras.models.load_model(model)
    
    rows = []
    for f in os.listdir(directory):
        file_type = imghdr.what(directory + '/' + f)
        if file_type in ('png', 'jpeg'):
            test_image = image.load_img(directory + '/' + f, target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = new_model.predict(test_image)
            result = np.argmax(result, axis=1)
#            training_set.class_indices
            if result[0] == 0:
                prediction = 'cat'
            elif result[0] == 1:
                prediction = 'dog'
            else:
                prediction = 'unknown_class'
        else:
            prediction = 'unsupported_file'
        rows.append([f, prediction])
    results_table = pd.DataFrame(rows, columns=["Image Name", "Prediction"])
    print(results_table)

if len(argv) > 1:
    model = argv[1]
    directory = argv[2]
else:
    model = 'saved_model/cnn_model_v1'
    directory = 'dataset/single_prediction'
    
cats_vs_dogs(model, directory)
    
