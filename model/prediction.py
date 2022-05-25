# -*- coding: utf-8 -*-
#calling the trained network for predictions

import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
from dataload import DataLoad
from prediction_feeder import PredictionDataFeeder
from stackandsplit import NormDict, StackedData
from IPython.display import display
import tensorflow.keras.backend as K


def display_mask(i):
    mask = val_preds[i]
    img = tf.keras.preprocessing.image.array_to_img(mask)
    display(img)

smooth = 1e-6

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    prod = y_true_f * y_pred_f
    intersection = tf.experimental.numpy.sum(prod)
    return (2. * intersection + smooth) / (tf.experimental.numpy.sum(y_true_f) + tf.experimental.numpy.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

if __name__ == '__main__':
    
    patients = []
    data = []
    masks = []
    data_folders = []
    
    parser = argparse.ArgumentParser(description = 
                                     '''Module to access the trained network and to make evaluation on new data.
                                          
                                         '''
                                         , formatter_class=RawTextHelpFormatter)
    parser.add_argument('basepath', 
                        type = str, 
                        help='Path to the working directory in which the data is contained')
    
    parser.add_argument('network', 
                        type = str, 
                        help='Path to the file which contains the trained network')
    
    parser.add_argument('img_size', 
                        type = tuple, 
                        help='Tuple containing the sizes of the images (sizes must be equal along X and Y axes). Sizes must match with those the nework was trained with')

    parser.add_argument('batch_size', 
                        type = int, 
                        help='Batch for the loading of the images to be evalued')    
    
    
    args = parser.parse_args()
    
    if args.basepath:
        basepath = Path(args.basepath) 
        

    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__' and not item.name == '.hypothesis':
                print(item.name)
                data_folders.append(item.name)
                path = basepath / '{}'.format(item.name)
                patients.append(path)
                
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__' and not item.name == '.hypothesis':
                for elem in item.iterdir():
                    if "Data" in elem.name:
                        data.append(elem)
                    elif "Segmentation" in elem.name:
                        masks.append(elem)
    
    data, data_array = DataLoad(data, masks)
    del(data)
    data = StackedData(data_array)
    del(data_array)
    
    model = tf.keras.models.load_model('{}'.format(args.network), custom_objects= {'dice_coef_loss' : dice_coef_loss, 'dice_coef': dice_coef})
    
    test_gen = PredictionDataFeeder(args.batch_size, args.img_size, data['features'])
    val_preds = model.predict(test_gen)
    
    for i in range(len(data['features'])):
        display_mask(i)
        img = data['features'][i]
        img = np.expand_dims(img, axis = -1)
        img = tf.keras.preprocessing.image.array_to_img(img)
        display(img)
