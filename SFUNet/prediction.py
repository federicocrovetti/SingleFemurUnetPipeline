#calling the trained network for predictions

import tensorflow as tf
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import csv
import argparse
from argparse import RawTextHelpFormatter
from sequential_feeder import PredictionFeeder
from SFUNet.utils.slicer_utils import PathExplorerSlicedDataset, SliceDatasetLoader, NIFTISlicesWriter 
from dataload import PathExplorer, DataLoad, MDTransfer, NIFTISingleSampleWriter 
from IPython.display import display
import tensorflow.keras.backend as K


smooth = 1e-6

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
    
    basepath = Path(args.basepath)
    new_folder_path = Path(args.new_folder_path)
        
    if new_folder_path.exists():
        pass
    else:
        new_folder_path.mkdir(exist_ok=True) 
           
    patients, data, masks, data_folders = PathExplorer(basepath)
        
    dataset, dataset_array = DataLoad(data, masks)
    print(dataset_array)
    print(dataset_array['features'].shape)
    test_set = PredictionFeeder(args.batch_size, dataset_array['features'])
    model = tf.keras.models.load_model('{}'.format(args.network), custom_objects= {'dice_coef_loss' : dice_coef_loss, 'dice_coef': dice_coef})
    val_preds = model.predict(test_set)
    
    if args.write_to_folder == True:
        new_folder_path = Path(args.new_folder_path)
        if new_folder_path.exists():
            pass
        else:
            new_folder_path.mkdir(exist_ok=True) 
            
        val_preds = (val_preds > 0.5).astype(np.uint8)
        preds = sitk.GetImageFromArray(val_preds)

        NIFTISingleSampleWriter(preds, 0000, new_folder_path)
    else:
        pass
