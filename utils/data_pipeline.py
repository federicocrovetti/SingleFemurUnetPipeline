import SimpleITK as sitk
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from pathlib import Path
from dataload import DataLoad, NIFTISampleWriter, NIFTISingleSampleWriter, MDTransfer
from square_complete import SquareComplete


def Halve(dataset, side, train = True):
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
                MDTransfer(dataset['features'][i], data)
                dataset_cropped['features'].append(data)
            else:
                data = dataset['features'][i][256 : 512, 0 : 512]
                MDTransfer(dataset['features'][i], data)
                dataset_cropped['features'].append(data)
        return dataset_cropped
    else:
        dataset_cropped =  {'features': [], 'labels':[]}
        for i in range(len(dataset['features'])):
            if side[i] == 0:
                data = dataset['features'][i][0 : 256, 0 : 512]
                MDTransfer(dataset['features'][i], data)
                labels = dataset['labels'][i][0 : 256, 0 : 512]
                MDTransfer(dataset['labels'][i], labels)
                dataset_cropped['features'].append(data)
                dataset_cropped['labels'].append(labels)
            elif side[i] == 1:
                data = dataset['features'][i][256 : 512, 0 : 512]
                MDTransfer(dataset['features'][i], data)
                labels = dataset['labels'][i][256 : 512, 0 : 512]
                MDTransfer(dataset['labels'][i], labels)
                dataset_cropped['features'].append(data)
                dataset_cropped['labels'].append(labels)
            else:
                raise Exception('Cannot recognize if the leg intended is the right or left one. Please insert L or R in the folder name accordingly.')
                
        return dataset_cropped


def BedRemoval(dataset, train = True):
    """
    
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
    if train == False:
        data = {'features' : []}
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
    else:
        data = {'features' : [], 'labels' : []}
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
            data['labels'].append(dataset['labels'][i])
            
    return data



def Thresholding(dataset, threshold, train = True):
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
            MDTransfer(dataset['features'][i], image)
            data['features'].append(image)
    else:
        data = {'features' : [], 'labels' : []}
        for i in range(len(dataset['features'])):
            image = thresh.Execute(dataset['features'][i])
            MDTransfer(dataset['features'][i], image)
            data['features'].append(image)
            data['labels'].append(dataset['labels'][i])
    
    return data


def BoundingBox(dataset):
    """
    
    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containins sitk.Image objects

    Returns
    -------
    bbox_list : list of lists for each image, for each slice, containing numpy arrays with: lower and upper bounderies along
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
            if any(dataset['features'][int(i)][:, :,j]): #any returns True if it sees 1    #ho cambiato labels con features
                crpfilter.Execute(dataset['features'][i][:, :, j], dataset['features'][i][:, :, j])    
                boundingbox = crpfilter.GetBoundingBox(1)#np.array(crpfilter.GetBoundingBox(1))
                
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


def Padding(data, compleat_measure, train = True):
    """

    Parameters
    ----------
    data : dict type object with 'features' and, optionally, 'labels' as keys containins sitk.Image objects
    compleat_measure : list containing the lists of bounding boxes for the objects in the dataset
    train : Str, optional. It determines if the function should be applied even on labels if those were 
    provided as part of the dataset object. The default is True.

    Returns
    -------
    pad_img and/or-not pad_labels : SimpleITK images padded with as many values, equal to -3000, as
                                    specified by 'compleat_measure'.

    """

    if train == True:
        pad = sitk.ConstantPadImageFilter()
        pad.SetGlobalDefaultDirectionTolerance(7.21e-1)
        pad.SetGlobalDefaultCoordinateTolerance(7.21e-1)
        pad.SetConstant(-3000)
        pad.SetPadUpperBound(compleat_measure)
        pad_img = pad.Execute(data['features'])
        pad_img.SetOrigin([-9.1640584e+01, -1.8851562e+02])
        pad.SetConstant(0)
        pad.SetPadUpperBound(compleat_measure)
        pad_labels = pad.Execute(data['labels'])
        pad_labels.SetOrigin([-9.1640584e+01, -1.8851562e+02])
        
        return pad_img, pad_labels
    else:
        pad = sitk.ConstantPadImageFilter()
        pad.SetGlobalDefaultDirectionTolerance(7.21e-1)
        pad.SetGlobalDefaultCoordinateTolerance(7.21e-1)
        pad.SetConstant(-3000)
        pad.SetPadUpperBound(compleat_measure)
        pad_img = pad.Execute(data['features'])
        pad_img.SetOrigin([-9.1640584e+01, -1.8851562e+02])
        
        return pad_img


def Crop(dataset, bbox_grouped, ID, new_folder_path, write_to_folder = True, train = True):
    """

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
    if write_to_folder == False and train == True:
        dataset_cropped =  {'features': [], 'labels':[]}
        for j in range(len(dataset['features'])):
            cropped = {'features' : [], 'labels' : []}
            crop_pad = {'features' : [], 'labels' : []}
            for i in range(len(bbox_grouped[j])):
                if bbox_grouped[j][i][0] == 7000:
                    pass
                else:
                    y_min = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] - 128
                    y_max = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] + 128
                    if y_min >= 0 and y_max <= 512:
                        data = dataset['features'][j][:, :, i][:, int(y_min) : int(y_max)]
                        labels = dataset['labels'][j][:, :, i][:, int(y_min) : int(y_max)] 
                    elif y_min >= 0 and y_max > 512:
                        data = dataset['features'][j][:, :, i][:, (y_min - y_max -512) : 512]
                        labels = dataset['labels'][j][:, :, i][:, (y_min - y_max -512) : 512]
                    elif y_min < 0 and y_max <= 512:
                        data = dataset['features'][j][:, :, i][:, 0 : (y_max + np.abs(y_min + 512))]
                        labels = dataset['labels'][j][:, :, i][:, 0 : (y_max + np.abs(y_min + 512))]
                    elif y_min < 0 and y_max > 512:
                        ValueError('The input image is over 512 pixels in size on the y axis')
                        
                        
                    data.SetOrigin([-9.1640584e+01, -1.8851562e+02])    
                    labels.SetOrigin([-9.1640584e+01, -1.8851562e+02])                  
                    cropped['features'].append(data)
                    cropped['labels'].append(labels)
                    
            sqcmplt = SquareComplete(cropped, (256,256))  
            for h in range(len(cropped['features'])):
                compleat_measure = sqcmplt[h]
                data = {key: val[h] for key, val in cropped.items()}
                pad_img, pad_labels = Padding(data, compleat_measure, train = True)
                crop_pad['features'].append(pad_img)
                crop_pad['labels'].append(pad_labels)
            join = sitk.JoinSeriesImageFilter()
            join.SetGlobalDefaultCoordinateTolerance(14.21e-1)
            crop_pad_volume = join.Execute([crop_pad['features'][k] for k in range(len(crop_pad['features']))])
            crop_pad_labels = join.Execute([crop_pad['labels'][k] for k in range(len(crop_pad['labels']))])
            MDTransfer(dataset['features'][j], crop_pad_volume)
            MDTransfer(dataset['labels'][j], crop_pad_labels)
            dataset_cropped['features'].append(crop_pad_volume)
            dataset_cropped['labels'].append(crop_pad_labels)
            
        return dataset_cropped
    
    elif write_to_folder == True and train == True:
        for j in range(len(dataset['features'])):
            cropped = {'features' : [], 'labels' : []}
            crop_pad = {'features' : [], 'labels' : []}
            for i in range(len(bbox_grouped[j])):
                if bbox_grouped[j][i][0] == 7000:
                    pass
                else:
                    y_min = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] - 128
                    y_max = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] + 128
                    if y_min >= 0 and y_max <= 512:
                        data = dataset['features'][j][:, :, i][:, int(y_min) : int(y_max)]
                        labels = dataset['labels'][j][:, :, i][:, int(y_min) : int(y_max)] 
                    elif y_min >= 0 and y_max > 512:
                        data = dataset['features'][j][:, :, i][:, (y_min - y_max -512) : 512]
                        labels = dataset['labels'][j][:, :, i][:, (y_min - y_max -512) : 512] 
                    elif y_min < 0 and y_max <= 512:
                        data = dataset['features'][j][:, :, i][:, 0 : (y_max + np.abs(y_min + 512))]
                        labels = dataset['labels'][j][:, :, i][:, 0 : (y_max + np.abs(y_min + 512))]
                    elif y_min < 0 and y_max > 512:
                        ValueError('The input image is over 512 pixels in size on the y axis')
                    
                    data.SetOrigin([-9.1640584e+01, -1.8851562e+02])    
                    labels.SetOrigin([-9.1640584e+01, -1.8851562e+02])               
                    cropped['features'].append(data)
                    cropped['labels'].append(labels)
           
            sqcmplt = SquareComplete(cropped, (256,256))  
            for h in range(len(cropped['features'])):
                compleat_measure = sqcmplt[h]
                data = {key: val[h] for key, val in cropped.items()}
                pad_img, pad_labels = Padding(data, compleat_measure, train = True)
                crop_pad['features'].append(pad_img)
                crop_pad['labels'].append(pad_labels)
            join = sitk.JoinSeriesImageFilter()
            join.SetGlobalDefaultCoordinateTolerance(14.21e-1)
            crop_pad_volume = join.Execute([crop_pad['features'][k] for k in range(len(crop_pad['features']))])
            crop_pad_labels = join.Execute([crop_pad['labels'][k] for k in range(len(crop_pad['labels']))])
            MDTransfer(dataset['features'][j], crop_pad_volume)
            MDTransfer(dataset['labels'][j], crop_pad_labels)
            
            NIFTISampleWriter(crop_pad_volume, crop_pad_labels, ID[j], new_folder_path)
        
        return
    
    elif write_to_folder == False and train == False:
        dataset_cropped =  {'features': []}
        for j in range(len(dataset['features'])):
            cropped = {'features' : []}
            crop_pad = {'features' : []}
            for i in range(len(bbox_grouped[j])):
                if bbox_grouped[j][i][0] == 7000:
                    pass
                else:
                    y_min = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] - 128
                    y_max = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] + 128
                    if y_min >= 0 and y_max <= 512:
                        data = dataset['features'][j][:, :, i][:, int(y_min) : int(y_max)]
                    elif y_min >= 0 and y_max > 512:
                        data = dataset['features'][j][:, :, i][:, (y_min - y_max -512) : 512]
                    elif y_min < 0 and y_max <= 512:
                        data = dataset['features'][j][:, :, i][:, 0 : (y_max + np.abs(y_min + 512))]
                    elif y_min < 0 and y_max > 512:
                        ValueError('The input image is over 512 pixels in size on the y axis')
                                                      
                    data.SetOrigin([-9.1640584e+01, -1.8851562e+02])               
                    cropped['features'].append(data)
                    
            sqcmplt = SquareComplete(cropped, (256,256))  
            for h in range(len(cropped['features'])):
                compleat_measure = sqcmplt[h]
                data = {key: val[h] for key, val in cropped.items()}
                pad_img = Padding(data, compleat_measure, train = False)
                crop_pad['features'].append(pad_img)
            join = sitk.JoinSeriesImageFilter()
            join.SetGlobalDefaultCoordinateTolerance(14.21e-1)
            crop_pad_volume = join.Execute([crop_pad['features'][k] for k in range(len(crop_pad['features']))])
            MDTransfer(dataset['features'][j], crop_pad_volume)
            dataset_cropped['features'].append(crop_pad_volume)
            
        return dataset_cropped
    
    elif write_to_folder == True and train == False:
        for j in range(len(dataset['features'])):
            cropped = {'features' : []}
            crop_pad = {'features' : []}
            for i in range(len(bbox_grouped[j])):
                if bbox_grouped[j][i][0] == 7000:
                    pass
                else:
                    y_min = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] - 128
                    y_max = ((bbox_grouped[j][i][3] - bbox_grouped[j][i][2])//2) + bbox_grouped[j][i][2] + 128
                    if y_min >= 0 and y_max <= 512:
                        data = dataset['features'][j][:, :, i][:, int(y_min) : int(y_max)]
                    elif y_min >= 0 and y_max > 512:
                        data = dataset['features'][j][:, :, i][:, (y_min - y_max -512) : 512]
                    elif y_min < 0 and y_max <= 512:
                        data = dataset['features'][j][:, :, i][:, 0 : (y_max + np.abs(y_min + 512))]
                    elif y_min < 0 and y_max > 512:
                        ValueError('The input image is over 512 pixels in size on the y axis')
                    
                    data.SetOrigin([-9.1640584e+01, -1.8851562e+02])               
                    cropped['features'].append(data)
           
            sqcmplt = SquareComplete(cropped, (256,256))  
            for h in range(len(cropped['features'])):
                compleat_measure = sqcmplt[h]
                data = {key: val[h] for key, val in cropped.items()}
                pad_img = Padding(data, compleat_measure, train = False)
                crop_pad['features'].append(pad_img)
            join = sitk.JoinSeriesImageFilter()
            join.SetGlobalDefaultCoordinateTolerance(14.21e-1)
            crop_pad_volume = join.Execute([crop_pad['features'][k] for k in range(len(crop_pad['features']))])
            MDTransfer(dataset['features'][j], crop_pad_volume)
            NIFTISingleSampleWriter(crop_pad_volume, ID[j], new_folder_path)
        
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
    
    parser.add_argument('write_to_folder',
                        metavar = 'write_to_folder',
                        type = str,
                        help = 'True if the data is to be written in target directory, False if not')
    
    parser.add_argument('train', 
                        metavar='train',
                        type = str, 
                        help='True when we have labels (training phase), False when we do not. The default is True.')
    
    
    args = parser.parse_args() 
    
    new_folder_path = Path(args.new_folder_path)
    if new_folder_path.exists():
        pass
    else:
        new_folder_path.mkdir(exist_ok=True) 
           
    
    basepath = Path(args.basepath)
    
    
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__' and not item.name == '.hypothesis' and not item.name == '_logdir_' and not item.name == 'Fedz':
                print(item.name)
                data_folders.append(item.name)
                path = basepath / '{}'.format(item.name)
                patients.append(path)
                
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__' and not item.name == '.hypothesis' and not item.name == '_logdir_' and not item.name == 'Fedz':
                for elem in item.iterdir():
                    if "Data" in elem.name:
                        data.append(elem)
                    elif "Segmentation" in elem.name:
                        masks.append(elem)
    
    ID = data_folders
    data, data_array= DataLoad(data, masks)  
    
    side = []
    for elem in data_folders:
        if 'R' in elem:
            side.append(0)
        elif 'L' in elem:
            side.append(1)
        else:
            del(side)
    dataset = BedRemoval(data, train=args.train)        
    halve_dst = Halve(dataset, side, train = args.data_with_masks)     
    
    thresh_dst = Thresholding(halve_dst, [args.low_end_threshold, args.high_end_threshold])
    label_sizes = BoundingBox(thresh_dst)
    Crop(halve_dst, label_sizes, ID, new_folder_path, write_to_folder = args.write_to_folder, train = args.train)