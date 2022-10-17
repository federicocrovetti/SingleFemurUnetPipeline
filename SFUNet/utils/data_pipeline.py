import SimpleITK as sitk
import csv
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from pathlib import Path
from SFUNet.utils.dataload import PathExplorer, DataLoad, NIFTISampleWriter, NIFTISingleSampleWriter, MDTransfer
from SFUNet.utils.square_complete import SquareComplete


def Halve(dataset, side, train = True):
    """
    This function takes a dict object with a list of sitk.Image(s) of shape (512,512,z) (both for features and labels)
    and return only the upper half third, where the side taken depends on the naming of the folder into which they were
    contained.
    
    Parameters
    ----------
    dataset : dict type object with 'features' and, optionally, 'labels' as keys containins sitk.Image objects
    side : list of binary ints 0 or 1, where 0 indicates the Right and 1 indicates the Left
    train : Str, optional. It determines if the function should be applied even on labels if those were 
    provided as part of the dataset object. The default is True.
    
    Returns
    -------
    dataset_cropped : dict type object with 'features' and, optionally, 'labels' as keys containing the first third of 
    the right or the left part of the volume provided (depending on which side is labeled in the folder name)
    
    """
    
    dataset_cropped =  {'features': []}
    if train:
        dataset_cropped["labels"] = []
    for i in range(round(len(dataset['features']))):
        if side[i] == 0:
            data = dataset['features'][i][0 : 256, 0 : 512]
            data = data[:,:, round(data.GetSize()[2]/3):512]
            MDTransfer(dataset['features'][i], data)
            dataset_cropped['features'].append(data)
            if train:
                labels = dataset['labels'][i][0 : 256, 0 : 512]
                labels = labels[:,:, round(labels.GetSize()[2]/3):512]
                MDTransfer(dataset['labels'][i], labels)
                dataset_cropped['labels'].append(labels)
        elif side[i] == 1:
            data = dataset['features'][i][256 : 512, 0 : 512]
            data = data[:,:,round(data.GetSize()[2]/3):512]
            MDTransfer(dataset['features'][i], data)
            dataset_cropped['features'].append(data)
            if train:
                labels = dataset['labels'][i][256 : 512, 0 : 512]
                labels = labels[:,:, round(labels.GetSize()[2]/3):512]
                MDTransfer(dataset['labels'][i], labels)
                dataset_cropped['labels'].append(labels)
        else:
            raise Exception('Cannot recognize if the leg intended is the right or left one. Please insert L or R in the folder name accordingly.')
    
    return dataset_cropped


def BedRemoval(dataset, train = True):
    """
    This function takes a dict object with a list of sitk.Image(s) and applies to the features an erosion with 
    fixed parameters radius = 4 and foreground value = 1 (since a binary thresholding was applied before),
    saving the major connected component.
    
    Parameters
    ----------
    dataset : dict type object with 'features' and, optionally, 'labels' as keys containins sitk.Image objects
    train : Str, optional. It determines if the function should be applied even on labels if those were 
        provided as part of the dataset object. The default is True.
    Returns
    -------
    data : dict type object with 'features' and, optionally, 'labels' as keys containing 
        only the right or the left part of the volume provided
    """
    thresh = sitk.BinaryThresholdImageFilter()
    thresh.SetInsideValue(1)
    thresh.SetOutsideValue(0)
    thresh.SetLowerThreshold(-400)
    thresh.SetUpperThreshold(10000)
    
    data = {'features' : []}
    if train:
        data["labels"] = []
    
    for i in range(len(dataset['features'])):
        image = thresh.Execute(dataset['features'][i])
        erode = sitk.BinaryErodeImageFilter()
        erode.SetForegroundValue(1)
        erode.SetKernelRadius(4)
        bin_eroded = erode.Execute(image)
        mask = sitk.MaskImageFilter()
        mask.SetOutsideValue(-3000)
        nobed_image = mask.Execute(dataset['features'][i], bin_eroded)
        MDTransfer(dataset['features'][i], nobed_image)
        data['features'].append(nobed_image)
        if train:
            data['labels'].append(dataset['labels'][i])
            
    return data



def Thresholding(dataset, threshold, train = True):
    """
    Given threshold parameters from the user, this function binarize the sitk.Image(s) contained in the dict object .
    
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
    data = {'features' : []}
    if train:
        data["labels"] = []
    for i in range(len(dataset['features'])):
        image = thresh.Execute(dataset['features'][i])
        MDTransfer(dataset['features'][i], image)
        data['features'].append(image)
        if train:
            data['labels'].append(dataset['labels'][i])
    
    return data


def BoundingBox(dataset):
    """
    Using sitk.LabelStatisticsImageFilter this function identifies the bounding box of the sitk.Image(s),
    calculate the center of it, and individuates a squared region of size (256,256), on the x,y plane,
    which contain the ROI (e.g. the femur).
    the region is returned in the form of a list of lists, were the foremost identifies each patient, while the 
    foremost each individual slice composing the image.
    
    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containins sitk.Image objects
    Returns
    -------
    bbox_list : list of lists for each image, for each slice, containing numpy arrays with: lower and upper boundaries along
                x and y directions and lower and upper boundaries for a patch, with size (256, 256), centered on the bounding
                box center
    """

    bbox_grouped =[]
    for i in range(len(dataset['features'])):
        crpfilter = sitk.LabelStatisticsImageFilter()
        crpfilter.SetGlobalDefaultDirectionTolerance(7.21e-1)
        crpfilter.SetGlobalDefaultCoordinateTolerance(7.21e-1)
        vol_box = []
        for j in range(int(dataset['features'][i].GetSize()[2])):
            if any(dataset['features'][int(i)][:, :,j]):
                crpfilter.Execute(dataset['features'][i][:, :, j], dataset['features'][i][:, :, j])    
                boundingbox = crpfilter.GetBoundingBox(1)
                
                y_min = round(((boundingbox[3] - boundingbox[2])//2) + boundingbox[2] - 128)
                y_max = round(((boundingbox[3] - boundingbox[2])//2) + boundingbox[2] + 128)
                
                if y_min >= 0 and y_max <= 512:
                    vol_box.append(np.array([boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[3],y_min, y_max]))
                elif y_min >= 0 and y_max > 512:
                    y_min = (y_min - y_max -512)
                    y_max = 512
                    vol_box.append(np.array([boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[3],y_min, y_max]))
                elif y_min < 0 and y_max <= 512:
                    y_min = 0
                    y_max = (y_max + np.abs(y_min + 512))
                    vol_box.append(np.array([boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[3],y_min, y_max]))
                elif y_min < 0 and y_max > 512:
                    ValueError('The input image is over 512 pixels in size on the y axis')
                
            else:
                vol_box.append([7000, 7000, 7000, 7000, 7000, 7000])
        bbox_grouped.append(vol_box)
        
    return bbox_grouped 


def Crop(dataset, bbox_grouped, ID, new_folder_path, write_to_folder = False, train = True):
    """
    Based on a list of lists containing the bounding boxes of each slice, this function crops each slice of the image
    and then joins them in a (256,256,z) sitk.Image.
    The result will be a (256,256,z) image of misaligned slices containing the femur.
    
    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containing the array form of the images and the labels
    label_sizes : list containing the lists of bounding boxes for the objects in the dataset
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
    dataset_cropped =  {'features': []}
    if train:
        dataset_cropped['labels'] = []
    
    for j in range(len(dataset['features'])):
        cropped = {'features' : []}
        if train:
            cropped['labels'] = []
        for i in range(len(bbox_grouped[j])):
            if bbox_grouped[j][i][0] == 7000:
                pass
            else:
                y_min = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] - 128
                y_max = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] + 128
                if y_min >= 0 and y_max <= 512:
                    data = dataset['features'][j][:, :, i][:, int(y_min) : int(y_max)]
                    if train:
                        labels = dataset['labels'][j][:, :, i][:, int(y_min) : int(y_max)] 
                elif y_min >= 0 and y_max > 512:
                    data = dataset['features'][j][:, :, i][:, (y_min - y_max -512) : 512]
                    if train:
                        labels = dataset['labels'][j][:, :, i][:, (y_min - y_max -512) : 512] 
                elif y_min < 0 and y_max <= 512:
                    data = dataset['features'][j][:, :, i][:, 0 : (y_max + np.abs(y_min + 512))]
                    if train:
                        labels = dataset['labels'][j][:, :, i][:, 0 : (y_max + np.abs(y_min + 512))]
                elif y_min < 0 and y_max > 512:
                    ValueError('The input image is over 512 pixels in size on the y axis')
                
                data.SetOrigin([-9.1640584e+01, -1.8851562e+02])
                cropped['features'].append(data)
                if train:
                    labels.SetOrigin([-9.1640584e+01, -1.8851562e+02])               
                    cropped['labels'].append(labels)
       
        join = sitk.JoinSeriesImageFilter()
        join.SetGlobalDefaultCoordinateTolerance(14.21e-1)
        crop_pad_volume = join.Execute([cropped['features'][k] for k in range(len(cropped['features']))])
        MDTransfer(dataset['features'][j], crop_pad_volume)
        if train:
            crop_pad_labels = join.Execute([cropped['labels'][k] for k in range(len(cropped['labels']))])
            MDTransfer(dataset['labels'][j], crop_pad_labels)
        if write_to_folder:
            NIFTISampleWriter(crop_pad_volume, crop_pad_labels, ID[j], new_folder_path)
            return
        else:
            if train:
                return crop_pad_volume, crop_pad_labels
            else:
                return crop_pad_volume


if __name__ == '__main__':
    
    
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
#CAMBIO    
    parser.add_argument('write_to_folder',
                        metavar = 'write_to_folder',
                        type = bool,
                        help = 'True if the data is to be written in target directory, False if not')
    
    parser.add_argument('train', 
                        metavar='train',
                        type = bool, 
                        help='True when we have labels (training phase), False when we do not. The default is True.')
#CAMBIO    
    parser.add_argument('--bbox_csv',
                        metavar = 'bbox_csv', 
                        type = str,
                        help='Optional argument that allows to individuate the path of the .txt file into which the bounding box coordinates will be written')
#NUOVO  
    parser.add_argument('--metadata_csv',
                        metavar = 'metadata_csv', 
                        type = str,
                        help='Optional argument that allows to write a .txt containing the metadata of original images in the format ([spacing x y z origin x y x])  where each row represents a volume image' )
    
    
    args = parser.parse_args() 
    
    new_folder_path = Path(args.new_folder_path)
    if new_folder_path.exists():
        pass
    else:
        new_folder_path.mkdir(exist_ok=True) 
           
    
    basepath = Path(args.basepath)
    
    patients, data_paths, masks_paths, data_folders = PathExplorer(basepath)
    
    ID = [[elem] for elem in data_folders]
    
    side = []
    for elem in data_folders:
        if 'R' in elem:
            side.append(0)
        elif 'L' in elem:
            side.append(1)
        else:
            del(side)
    
    for i in range(len(data_folders)):
        data, data_array= DataLoad(data_paths[i], masks_paths[i])
        del(data_array)
        dataset = BedRemoval(data, train=args.train)        
        halve_dst = Halve(dataset, side, train = args.train)  
        
        thresh_dst = Thresholding(halve_dst, [args.low_end_threshold, args.high_end_threshold], train = args.train)
        label_sizes = BoundingBox(thresh_dst)
        Crop(halve_dst, label_sizes, ID[i], new_folder_path, write_to_folder = args.write_to_folder, train = args.train)
        
        if args.bbox_csv is not None:
            with open(Path('{}'.format(args.bbox_csv)), 'a', newline='', encoding='UTF8') as f:
                writer = csv.writer(f, delimiter=',')
                for i in range(len(label_sizes)):
                    for j in range(len(label_sizes[i])):
                        writer.writerow(label_sizes[i][j])
                       
        if args.metadata_csv is not None:
            with open(Path('{}'.format(args.metadata_csv)), 'a', newline='', encoding='UTF8') as f:
                writer = csv.writer(f, delimiter=',')
                for i in range(len(data['features'])):
                    writer.writerow((dataset['features'][i].GetSpacing()[0], dataset['features'][i].GetSpacing()[1], dataset['features'][i].GetSpacing()[2],
                                      dataset['features'][i].GetOrigin()[0], dataset['features'][i].GetOrigin()[1], dataset['features'][i].GetOrigin()[2]))
            
