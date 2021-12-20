import SimpleITK as sitk
from pathlib import Path
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from dataload import DataLoad, NIFTISampleWriter


def BoundingBox(dataset):
    """
    

    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containins sitk.Image objects

    Returns
    -------
    bbox_list : list containing the sizes of the minimum bounding boxes containing the labels
                in the form [xmin, xmax, ymin, ymax, zmin, zmax]

    """
    bbox_list = []
    for i in range(len(dataset['features'])):
        crpfilter = sitk.LabelStatisticsImageFilter()
        crpfilter.SetGlobalDefaultDirectionTolerance(7.21e-1)
        crpfilter.SetGlobalDefaultCoordinateTolerance(7.21e-1)
        crpfilter.Execute(dataset['features'][i], dataset['labels'][i])
        boundingbox = np.array(crpfilter.GetBoundingBox(1))
        bbox_list.append(boundingbox)
    return bbox_list


def Crop(dataset, label_sizes, ID, new_folder_path):
    """
    

    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containins sitk.Image objects
    label_sizes : list containing the bounding boxes of the objects in dataset
    ID : list containing the names with which the new samples will be named after
    new_folder_path : pathlib path object to the new folder inside which the cropped images will be written into

    Returns
    -------
    dataset_cropped : dict type object with 'features' and 'labels' as keys containins sitk.Image objects which are 
                      the cropped images and labels

    """
    
    for i in range(len(dataset['features'])):
        data = dataset['features'][i][label_sizes[i][0] : label_sizes[i][1], label_sizes[i][2] : label_sizes[i][3],
                                          label_sizes[i][4] : label_sizes[i][5]]
        labels = dataset['labels'][i][label_sizes[i][0] : label_sizes[i][1], label_sizes[i][2] : label_sizes[i][3],
                                          label_sizes[i][4] : label_sizes[i][5]]
        NIFTISampleWriter(data, ID[i], new_folder_path, image_and_mask = 0, volume_mask = labels)
        
        return 
    


if __name__ == '__main__':
    patients = []
    data = []
    masks = []
    data_folders = []
    
    parser = argparse.ArgumentParser(description = 
                                     '''Module for the cropping of the dataset capable of reading DICOM and NIFTI images.
                                     Uses the SimpleITK class LabelStatisticsImageFilter and its method GetBoundingBox
                                     for the automatic detection of the bounding box delimiting the labeled zone.
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
    
    
    args = parser.parse_args()

    
    basepath = Path(args.basepath) 
    
    new_folder_path = Path(args.new_folder_path)
    if new_folder_path.exists():
        pass
    else:
        new_folder_path.mkdir(exist_ok=True) 
           

    
    
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
    dataset, dataset_array= DataLoad(data, masks)

    label_sizes = BoundingBox(dataset)
    Crop(dataset, label_sizes, ID, new_folder_path, write_to_folder = True)
   
                
    
