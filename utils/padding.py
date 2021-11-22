#padding

import SimpleITK as sitk
from pathlib import Path
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from dataload import DataLoad, DICOMSampleWriter, NIFTISampleWriter


def SquareComplete(dataset, req_size):
    """
    

    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containins sitk.Image objects
    req_size : (int, int) or (int, int, int) tuple with the minimum desired size the image is required 
    to be after the padding along x, y and, optionally z, axis.

    Returns
    -------
    pad_sizes : list of lists with the necessary padding sizes

    """
    if len(req_size) == 2:
        pad_sizes = []
        for i in range(len(dataset['features'])):
            size = dataset['features'][i].GetSize()
            pad_size = (req_size[0]-size[0], req_size[1]-size[1], 0)
            pad_size = list(pad_size)
            pad_sizes.append(pad_size)
        return pad_sizes
    
    elif len(req_size) == 3:
        pad_sizes = []
        for i in range(len(dataset['features'])):
            size = dataset['features'][i].GetSize()
            pad_size = (req_size[0]-size[0], req_size[1]-size[1], req_size[2]-size[2])
            pad_size = list(pad_size)
            pad_sizes.append(pad_size)
        return pad_sizes
    


def Padding(dataset, ID, new_path, up_bound, constant = None):
    """
    

    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containins sitk.Image objects
    ID : list containing the names with which the new samples will be named after
    new_folder_path : pathlib path object to the new folder inside which the cropped images will be written into
    up_bound : list containing the padding in the upper bound
    low_bound : list containing the padding in the lower bound. The default is [0, 0, 0].
    constant : constant value for the padding. The default is None.


    """
    if up_bound is not list:
        up_bound = list(up_bound)
    if new_path.exists():
        pass
    else:
        new_path.mkdir(exist_ok=True)
    
    for i in range(len(dataset['features'])):
        pad = sitk.ConstantPadImageFilter()
        print(type(up_bound))
        pad.SetPadUpperBound(up_bound[i])
        pad_img = pad.Execute(dataset['features'][i])
        pad_labels = pad.Execute(dataset['labels'][i])
        NIFTISampleWriter(pad_img, ID[i], new_path, image_and_mask = 0, volume_mask = pad_labels)
    return 



if __name__ == '__main__':
    patients = []
    data = []
    masks = []
    data_folders = []
    
    parser = argparse.ArgumentParser(description = 
                                     '''Module for the cropping of the dataset capable of reading DICOM and NIFTI images.
                                     Takes the images from the desired directory as long as it has the required directory structure.
                                     This module make use of the SimpleITK class ConstantPadImageFilter to apply padding to the images. 
                                     Takes the images from the desired directory as long as it has the required directory structure,
                                     and writes the cropped images in the target directory with this structure:
                                     
                                     DATA  
                                         |
                                         Sample1
                                             |
                                             Images
                                                 |
                                                     >  ----
                                             |
                                             Labels
                                                 |
                                                     >  ----
                                          |
                                          Sample2
                                          .
                                          .
                                          .
                                          
                                         '''
                                         , formatter_class=RawTextHelpFormatter)
    parser.add_argument('basepath', 
                        metavar='basepath',
                        type = str, 
                        help='Path to the working directory in which the data is contained')
    
    parser.add_argument('new_folder_path',
                        metavar = 'new_folder_path',
                        type = str,
                        help = 'Path to the folder onto which the cropped samples will be written')
    
    parser.add_argument('reqsizex', 
                        metavar = 'requiredsizeX', 
                        type = int, #tuple,
                        help='Tuple which specifies the required dimensions for the image to have along X axis')
    
    parser.add_argument('reqsizey', 
                        metavar = 'requiredsizeY', 
                        type = int, #tuple,
                        help='Tuple which specifies the required dimensions for the image to havealong Y axis')
    
    
    args = parser.parse_args()


    basepath = Path(args.basepath) 
    new_folder_path = Path(args.new_folder_path)
    req_size = (args.reqsizex, args.reqsizey)
    if new_folder_path.exists():
        pass
    else:
        new_folder_path.mkdir(exist_ok=True)
        
    
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__':
                print(item.name)
                data_folders.append(item.name)
                path = basepath / '{}'.format(item.name)
                patients.append(path)
                
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__':
                for elem in item.iterdir():
                    if "Data" in elem.name:
                        data.append(elem)
                    elif "Segmentation" in elem.name:
                        masks.append(elem)
    
    ID = data_folders
    dataset, dataset_array= DataLoad(data, masks)
    
    pad_sizes = SquareComplete(dataset, req_size)
    Padding(dataset, ID, new_folder_path, pad_sizes, constant = np.min(dataset['features'][0]))
