import SimpleITK as sitk
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from pathlib import Path
from dataload import DataLoad, NIFTISampleWriter

def halve(dataset, side, train = True):
    """
    

    Parameters
    ----------
    dataset : dict type object with 'features' and, optionally, 'labels' as keys containins sitk.Image objects
    side : list of binary ints 0 or 1, where 0 indicates the Right and 1 indicates the Left
    train : Str, optional. It determines if the function should be applied even on labels if those were 
    provided as part of the dataset object. The default is True.

    Returns
    -------
    dataset_cropped : dict type object with 'features' and, optionally, 'labels' as keys containing 
    only the right or the left part of the volume provided

    """
    if train == False:
        dataset_cropped =  {'features': []}
        for i in range(len(dataset['features'])):
            if side[i] == 0:
                data = dataset['features'][i][0 : 256, 0 : 512]
                dataset_cropped['features'].append(data)
            else:
                data = dataset['features'][i][256 : 512, 0 : 512]
                dataset_cropped['features'].append(data)
        return dataset_cropped
    else:
        dataset_cropped =  {'features': [], 'labels':[]}
        for i in range(len(dataset['features'])):
            if side[i] == 0:
                data = dataset['features'][i][0 : 256, 0 : 512]
                labels = dataset['labels'][i][0 : 256, 0 : 512]
                dataset_cropped['features'].append(data)
                dataset_cropped['labels'].append(labels)
            else:
                data = dataset['features'][i][256 : 512, 0 : 512]
                labels = dataset['labels'][i][256 : 512, 0 : 512]
                dataset_cropped['features'].append(data)
                dataset_cropped['labels'].append(labels)
        return dataset_cropped
        
        
        def thresholding(dataset, threshold, train = True):
    """
    

    Parameters
    ----------
    dataset : dict type object with 'features' and, optionally, 'labels' as keys containins sitk.Image objects
    threshold : list object containing the interval for thresholding
    train : Str, optional. It determines if the function returns a dict with a second key containing 
    the labels only if they were part of the dataset object. The default is True.

    Returns
    -------
    data : dict type object with 'features' and, optionally, 'labels' as keys containing
    binarized thresholded sitk.Image objects

    """
    thresh = sitk.BinaryThresholdImageFilter()
    thresh.SetInsideValue(1)
    thresh.SetOutsideValue(0)
    thresh.SetLowerThreshold(threshold[0])
    thresh.SetUpperThreshold(threshold[1])
    if train == False:
        data = {'features' : []}
        for i in range(len(dataset['features'])):
            image = thresh.Execute(dataset['features'][i])
            data['features'].append(image)
    else:
        data = {'features' : [], 'labels' : []}
        for i in range(len(dataset['features'])):
            image = thresh.Execute(dataset['features'][i])
            data['features'].append(image)
            data['labels'].append(dataset['labels'])
    
    return data


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
        crpfilter = sitk.LabelShapeStatisticsImageFilter()
        crpfilter.SetGlobalDefaultDirectionTolerance(7.21e-1)
        crpfilter.SetGlobalDefaultCoordinateTolerance(7.21e-1)
        crpfilter.Execute(dataset['features'][i])
        boundingbox = np.array(crpfilter.GetBoundingBox(1))
        bbox_list.append(boundingbox)
    return bbox_list
    
    def Crop(dataset, label_sizes, ID, new_folder_path, write_to_folder = None):
    """
    

    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containins sitk.Image objects
    label_sizes : list containing the bounding boxes of the objects in dataset
    ID : list containing the names with which the new samples will be named after
    new_folder_path : pathlib path object to the new folder inside which the cropped images will be written into
    write_to_folder : if FALSE the function won't write the cropped images into the folder at 'new_folder_path' but
                        will return a dict type object with 'features' and 'labels' as keys containins sitk.Image objects
                        
                      if TRUE doesn't return anything and write directly the images into the specified folder

    Returns
    -------
    dataset_cropped : dict type object with 'features' and 'labels' as keys containins sitk.Image objects which are 
                      the cropped images and labels

    """
    if write_to_folder == False:
        dataset_cropped =  {'features': [], 'labels':[]}
        for i in range(len(dataset['features'])):
            data = dataset['features'][i][label_sizes[i][0] : label_sizes[i][1], label_sizes[i][2] : label_sizes[i][3],
                                          label_sizes[i][4] : label_sizes[i][5]]
            #data = data[0 : data.GetSize()[0], 0 : data.GetSize()[1], 0 : (data.GetSize()[2]/3)]
            labels = dataset['labels'][i][label_sizes[i][0] : label_sizes[i][1], label_sizes[i][2] : label_sizes[i][3],
                                          label_sizes[i][4] : label_sizes[i][5]]
            #labels = labels[0 : labels.GetSize()[0], 0 : labels.GetSize()[1], 0 : (data.GetSize()[2]/3)]
            dataset_cropped['features'].append(data)
            dataset_cropped['labels'].append(labels)
            
        return dataset_cropped
    else:
        for i in range(len(dataset['features'])):
            data = dataset['features'][i][label_sizes[i][0] : label_sizes[i][1], label_sizes[i][2] : label_sizes[i][3],
                                          label_sizes[i][4] : label_sizes[i][5]]
            #data = data[0 : data.GetSize()[0], 0 : data.GetSize()[1], 0 : (data.GetSize()[2]/3)]
            labels = dataset['labels'][i][label_sizes[i][0] : label_sizes[i][1], label_sizes[i][2] : label_sizes[i][3],
                                          label_sizes[i][4] : label_sizes[i][5]]
            #labels = labels[0 : labels.GetSize()[0], 0 : labels.GetSize()[1], 0 : (data.GetSize()[2]/3)]
            
            NIFTISampleWriter(data, labels, ID[i], new_folder_path)
        
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
    
    parser.add_argument('low_end_threshold',
                        metavar = 'low_end_threshold',
                        type = int,
                        help = 'low end threshold value')
    
    parser.add_argument('high_end_threshold',
                        metavar = 'high_end_threshold',
                        type = int,
                        help = 'high end threshold value')
    
    parser.add_argument('data_with_masks',
                        metavar = 'data_with_masks',
                        type = str,
                        help = 'Data is provided with masks or not. True if it is the case, False if not')
    
    
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
    del(data_array)
    
    side = []
    for elem in data_folders:
        if 'R' in elem:
            side.append(0)
        elif 'L' in elem:
            side.append(1)
        else:
            del(side)
            
    halve_dst = halve(data, side, train = args.data_with_masks)
    del(data)        
    
    thresh_dst = thresholding(halve_dst, [args.low_end_threshold, args.high_end_treshold])
    label_sizes = BoundingBox(dataset)
    Crop(dataset, label_sizes, ID, new_folder_path, write_to_folder = True)
   
