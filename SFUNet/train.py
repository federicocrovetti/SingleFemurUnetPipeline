#training

import numpy as np
from pathlib import Path
from SFUNet.utils.dataload import DataLoad 
import tensorflow as tf
from SFUNet.model.feeder import ImageFeeder
from SFUNet.model.model2D import model2D
from SFUNet.utils.slicer_utils import PathExplorerSlicedDataset
import argparse
from argparse import RawTextHelpFormatter


def dice_coef(y_true, y_pred):
    """
    This function calculate the Dice coefficient of the predicted labels with respect to 
    the ground truth.
    
    Parameters
    ----------
    y_true : numpy ndarray containing the values of the label
    y_pred : numpy ndarray containing the predicted values for the label

    Returns
    -------
    Floating number,contained in the interval [0,1], representing the dice coefficient

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    prod = y_true_f * y_pred_f
    intersection = tf.experimental.numpy.sum(prod)
    return (2. * intersection + smooth) / (tf.experimental.numpy.sum(y_true_f) + tf.experimental.numpy.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    Calculation of the Dice Coefficient Loss

    Parameters
    ----------
    y_true : numpy ndarray containing the values of the label
    y_pred : numpy ndarray containing the predicted values for the label

    Returns
    -------
    Floating number,contained in the interval [0,1], representing the Dice Coefficient Loss for the network

    """
    return 1 - dice_coef(y_true, y_pred)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 
                                     '''Module for the training of 'model2D' Unet-like architecture. The user is to give the dataset in the required
                                     form, the sizes of the 2D images (which are required to be squared) and is free to choose the following preferences
                                     and parameters:
                                         - Basepath : path to the parent folder containing Train-Val-Test data;
                                         - epochs : the number of epochs for the training;
                                         - checkpoint_filepath : the path for the creation of the file containing the callbacks;
                                         - the path for the creation of the file where the trained model will be saved.
                                         
                                         '''
                                         , formatter_class=RawTextHelpFormatter)
    parser.add_argument('basepath', 
                        type = str, 
                        help='Path to the working directory in which the data is contained')
    
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
        
    
    trainfeat, trainlab, valfeat, vallab, testfeat, testlab = PathExplorerSlicedDataset(basepath)
    
    train_set = ImageFeeder(args.batch_size, trainfeat, trainlab)
    val_set = ImageFeeder(args.batch_size, valfeat, vallab)
    
    tf.keras.backend.clear_session()
    model = model2D((256, 256), 1)
    
    import tensorflow.keras.backend as K
    K.clear_session()
    lr = 1e-3
    smooth = 1e-6

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=dice_coef_loss,  metrics=[dice_coef]) 
    
    checkpoint_filepath = Path(args.ckptfile)
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    save_weights_only=True, monitor='val_dice_coef',
    mode='max', save_best_only=True)
    ]
    
    model.fit(train_set, epochs=args.epochs, validation_data = val_set, callbacks=callbacks, shuffle=False) 
    model.load_weights(checkpoint_filepath)
    model.save('{}'.format(args.trainedmod))
