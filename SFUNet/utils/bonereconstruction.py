#Bone and label reconstruction

import SimpleITK as sitk
import csv
from itertools import islice
import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from SFUNet.utils.dataload import DataLoad, NIFTISampleWriter, NIFTISingleSampleWriter, MDTransfer

def Reconstruction(data, ID, boundingbox, new_folder_path, train = False):
    """
    
    Parameters
    ----------
    data : dict type object with 'features' and 'labels' as keys containins sitk.Image objects
    ID : string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    boundingbox : txt file containing the bounding boxes and the lower and upper coordinates on y-axis where the patch was extracted
                    from the original image
    new_folder_path : the parent folder for the new data


    """
    if train == False:
        for i in range(len(data['features'])):
            print(data['features'][i].GetSize())
        for i in range(len(data['features'])):
            reconstructed = {'features' : []}
            for j in range(len(boundingbox[i])):
            
                cut_feat = data['features'][i][:, :, j][:,:]
                rec_slice_feat = sitk.Image(256, 512, sitk.sitkInt16)
                MDTransfer(cut_feat, rec_slice_feat)
                rec_slice_feat[0 : 256, int(boundingbox[i][j][4]) : int(boundingbox[i][j][5])] = cut_feat #, y_min : y_max] = cut_feat
                rec_slice_feat.SetOrigin([-9.1640584e+01, -1.8851562e+02])
       
                reconstructed['features'].append(rec_slice_feat)
            
            join = sitk.JoinSeriesImageFilter()
            join.SetGlobalDefaultCoordinateTolerance(14.21e-1)
            recon_volume = join.Execute([reconstructed['features'][k] for k in range(len(reconstructed['features']))])
            MDTransfer(data['features'][i], recon_volume)
        
            NIFTISingleSampleWriter(recon_volume, recon_labels, ID[i], new_folder_path)
        
    else:
        for i in range(len(data['features'])):
            print(data['features'][i].GetSize())
        for i in range(len(data['features'])):
            reconstructed = {'features' : [], 'labels' : []}
            for j in range(len(boundingbox[i])):
            
                cut_feat = data['features'][i][:, :, j][:,:]
                rec_slice_feat = sitk.Image(256, 512, sitk.sitkInt16)
                MDTransfer(cut_feat, rec_slice_feat)
                rec_slice_feat[0 : 256, int(boundingbox[i][j][4]) : int(boundingbox[i][j][5])] = cut_feat #, y_min : y_max] = cut_feat
                rec_slice_feat.SetOrigin([-9.1640584e+01, -1.8851562e+02])
        
                cut_lab = data['labels'][i][:, :, j][:,:]
                rec_slice_lab = sitk.Image(256, 512, sitk.sitkInt16)
                MDTransfer(cut_lab, rec_slice_lab)
                rec_slice_lab[0 : 256, int(boundingbox[i][j][4]) : int(boundingbox[i][j][5])] = cut_lab# y_min : y_max] = cut_lab
                rec_slice_lab.SetOrigin([-9.1640584e+01, -1.8851562e+02])
                reconstructed['features'].append(rec_slice_feat)
                reconstructed['labels'].append(rec_slice_lab)
            
            join = sitk.JoinSeriesImageFilter()
            join.SetGlobalDefaultCoordinateTolerance(14.21e-1)
            recon_volume = join.Execute([reconstructed['features'][k] for k in range(len(reconstructed['features']))])
            recon_labels = join.Execute([reconstructed['labels'][k] for k in range(len(reconstructed['labels']))])
            MDTransfer(data['features'][i], recon_volume)
            MDTransfer(data['labels'][i], recon_labels)
        
            NIFTISampleWriter(recon_volume, recon_labels, ID[i], new_folder_path)
        
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
    
    parser.add_argument('train', 
                        metavar='train',
                        type = bool, 
                        help='True when we have labels (training phase), False when we do not. The default is True.')
    
    parser.add_argument('csv_path', 
                        metavar='csv_path',
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
    dataset, dataset_array = DataLoad(data, masks)
    
    img_sizes = [dataset['features'][i].GetSize()[2] for i in range(len(dataset['features']))]
    bounding_box_list = []
    with open(Path('{}'.format(args.csv_path)), 'r', newline='', encoding='UTF8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            bounding_box_list.append(row)
 
    
    iterator = iter(bounding_box_list)
    bounding_box_grouped = [list(islice(iterator, elem))
          for elem in img_sizes]
                
    Reconstruction(dataset, ID, bounding_box_grouped, new_folder_path, train = args.train)

