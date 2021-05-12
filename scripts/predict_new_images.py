#!/usr/bin/python

import tensorflow as tf
import imghdr
import numpy as np
from keras.preprocessing import image
import os
import pandas as pd
from sys import argv

class PredictImages:
    def __init__(self):
        self.default_model = tf.keras.models.load_model('../saved_model/cnn_model_v1')
        self.images_loc = '../dataset/single_prediction'
        pass
    
    def load_saved_model(self, model_direc):
        self.model = tf.keras.models.load_model(model_direc)
        return self.model
    
    def new_predictions(self, images_direc = None, model = None):
        if model is None:
            model = self.default_model
        if images_direc is None:
            images_direc = self.images_loc
        self.rows = []
        for f in os.listdir(images_direc):
            self.file_type = imghdr.what(images_direc + '/' + f)
            if self.file_type in ('png', 'jpeg'):
                self.test_image = image.load_img(images_direc + '/' + f, target_size = (64, 64))
                self.test_image = image.img_to_array(self.test_image)
                self.test_image = np.expand_dims(self.test_image, axis = 0)
                self.result = model.predict(self.test_image)
                self.result = np.argmax(self.result, axis=1)
                if self.result[0] == 0:
                    self.prediction = 'cat'
                elif self.result[0] == 1:
                    self.prediction = 'dog'
                else:
                    self.prediction = 'unknown_class'
            else:
                self.prediction = 'unsupported_file'
            self.rows.append([f, self.prediction])
        self.results_table = pd.DataFrame(self.rows, columns=["Image Name", "Prediction"])
        print(self.results_table)
        
def main():
    predict = PredictImages()
    predict.new_predictions()

if __name__ == "__main__":
    main()
    
