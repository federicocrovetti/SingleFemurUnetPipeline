# -*- coding: utf-8 -*-
#training

import numpy as np
from pathlib import Path
from utils.dataload import DataLoad 
import tensorflow as tf
from model.feeder import ImageFeeder
from model.model2D import model2D
import argparse
from argparse import RawTextHelpFormatter


if __name__ == '__main__':
    
    patients = []
    data = []
    masks = []
    data_folders = []
    
    parser = argparse.ArgumentParser(description = 
                                     '''Module for the training of 'model2D' Unet-like architecture. The user is to give the dataset in the required
                                     form, the sizes of the 2D images (which are required to be squared) and is free to choose the following preferences
                                     and parameters:
                                         - x1, x2, x3 : the percentual splittings for the stacked dataset into train, validation and test sets;
                                         - epochs : the number of epochs for the training;
                                         - checkpoint_filepath : the path for the creation of the file containing the callbacks;
                                         - the path for the creation of the file where the trained model will be saved
                                         
                                         '''
                                         , formatter_class=RawTextHelpFormatter)
    parser.add_argument('basepath', 
                        type = str, 
                        help='Path to the working directory in which the data is contained')
    
    parser.add_argument('img_size', 
                        type = tuple, 
                        help='Tuple containing the sizes of the images (sizes must be equal along X and Y axes)')
    
    parser.add_argument('trainperc', 
                        type = float, 
                        help='Percentage of the total data to be used for the training')
    
    parser.add_argument('validperc', 
                        type = float, 
                        help='Percentage of the total data to be used for the model validation')
    
    parser.add_argument('testperc', 
                        type = float, 
                        help='Percentage of the total data to be used for the testing')
    
    parser.add_argument('batch_size', 
                        type = int, 
                        help='Batch size for the training')
    
    parser.add_argument('epochs', 
                        type = int, 
                        help='Number of epochs for the training process')
    
    
    parser.add_argument('ckptfile',
                        type = str,
                        help='Path for the file containing the keras checkponts')
    
    parser.add_argument('trainedmod',
                        type = str,
                        help='Path to the new directory onto which the resampled images will be written')
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
    
    
    dataset, dataset_array = DataLoad(data, masks)
    del dataset
    datatrain, dataval, datatest = Split(dataset_array, args.trainperc, args.valperc, args.testperc)
    del dataset_array

    train_dst = ImageFeeder(args.batch_size, args.img_size, datatrain['features'], datatrain['labels'])
    val_dst = ImageFeeder(args.batch_size, args.img_size, dataval['features'], dataval['labels'])
    
    tf.keras.backend.clear_session()
    model = model2D(args.img_size, 1)
    model.summary()
    import tensorflow.keras.backend as K
    
    lr = 1e-3
    smooth = 1e-6

    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        prod = y_true_f * y_pred_f
        intersection = tf.experimental.numpy.sum(prod)
        return (2. * intersection + smooth) / (tf.experimental.numpy.sum(y_true_f) + tf.experimental.numpy.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=dice_coef_loss,  metrics=[dice_coef]) 
    
    checkpoint_filepath = Path(args.ckptfile)
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    save_weights_only=True, monitor='val_dice_coef',
    mode='max', save_best_only=True)
    ]
    
    model.fit(train_dst, epochs=args.epochs, validation_data = val_dst, callbacks=callbacks) 
    model.save('{}'.format(args.trainedmod))


