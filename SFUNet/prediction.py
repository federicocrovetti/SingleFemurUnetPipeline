# -*- coding: utf-8 -*-
#calling the trained network for predictions

import tensorflow as tf
import numpy as np
from pathlib import Path
import csv
import argparse
from argparse import RawTextHelpFormatter
from SFUNet.utils.dataload import DataLoad, NIFTISingleSampleWriter
from SFUNet.model.prediction_feeder import PredictionDataFeeder
from SFUNet.utils.stackandsplit import NormDict, StackedData, Split
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
    
    parser.add_argument('write_to_folder', 
                        type = bool, 
                        help='The resulting SimpleITK image is written or not to the indicated path')

    parser.add_argument('batch_size', 
                        type = int, 
                        help='Batch for the loading of the images to be evalued')
    
    parser.add_argument('--new_folder_path', 
                        type = str, 
                        help='.')
    
    
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
    
    ID = data_folders
    
    data, data_array = DataLoad(data, masks)
    data = StackedData(data_array)
    del(data_array)
    
    model = tf.keras.models.load_model('{}'.format(args.network), custom_objects= {'dice_coef_loss' : dice_coef_loss, 'dice_coef': dice_coef})
    
    test_gen = PredictionDataFeeder(args.batch_size, (256,256), data['features'])
    val_preds = model.predict(test_gen)
    
    if args.write_to_folder == True:
        new_folder_path = Path(args.new_folder_path)
        if new_folder_path.exists():
            pass
        else:
            new_folder_path.mkdir(exist_ok=True) 
           
        preds = val_preds[:,:,:,0]
        preds = sitk.GetImageFromArray(preds)
        NIFTISingleSampleWriter(preds, ID, new_folder_path)
        
        #metadata of the original image volume
        with open(new_folder_path, 'w', newline='', encoding='UTF8') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(len(data['features'])):
                writer.writerow(data['features'][i].GetSpacing()[0], data['features'][i].GetSpacing()[1], data['features'][i].GetSpacing()[2],
                                data['features'][i].GetOrigin()[0], data['features'][i].GetOrigin()[1], data['features'][i].GetOrigin()[2])
    else:
        pass
    
    
