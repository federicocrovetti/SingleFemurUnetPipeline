#Bone and label reconstruction

import SimpleITK as sitk
import csv
from itertools import islice
import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from SFUNet.utils.dataload import PathExplorer, DataLoad, NIFTISampleWriter, NIFTISingleSampleWriter, MDTransfer


def ReconstructionExtMetadata(data, ID, boundingbox, new_folder_path, md, train = False):
    """
    This function, by means of txt files containing the original metadata of the images and the bounding boxes,
    takes a misaligned (256,256,z) image and rewrite it in a (256,512,z) image in the destination folder.
    The slices contained in the input image will be pasted at the correct location in the new blank image,
    reconstruting the original shape of the femur.
    
    Parameters
    ----------
    data : dict type object with 'features' and 'labels' as keys containins sitk.Image objects
    ID : string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    boundingbox : list containing the bounding boxes and the lower and upper coordinates on y-axis where the patch was extracted
                    from the original image
    new_folder_path : the parent folder for the new data
    md : list containing the metadata of the original image (spacing x, y, z, origin x, y, z)
    """
    reconstructed = {'features' : []}
    if train:
        reconstructed['labels'] = []
    
    for i in range(len(data['features'])):
        reconstructed = {'features' : []}
        if train:
            reconstructed['labels'] = []
        for j in range(len(boundingbox[i])):
            cut_feat = data['features'][i][:, :, j][:,:]
            rec_slice_feat = sitk.Image(256, 512, sitk.sitkInt16)
            MDTransfer(cut_feat, rec_slice_feat)
            rec_slice_feat[0 : 256, int(boundingbox[0][j][4]) : int(boundingbox[0][j][5])] = cut_feat #, y_min : y_max] = cut_feat
            rec_slice_feat.SetOrigin([-9.1640584e+01, -1.8851562e+02])
            reconstructed['features'].append(rec_slice_feat)
            if train:
                cut_lab = data['labels'][i][:, :, j][:,:]
                rec_slice_lab = sitk.Image(256, 512, sitk.sitkInt16)
                MDTransfer(cut_lab, rec_slice_lab)
                rec_slice_lab[0 : 256, int(boundingbox[0][j][4]) : int(boundingbox[0][j][5])] = cut_lab# y_min : y_max] = cut_lab
                rec_slice_lab.SetOrigin([-9.1640584e+01, -1.8851562e+02])
                reconstructed['labels'].append(rec_slice_lab)
        
        join = sitk.JoinSeriesImageFilter()
        join.SetGlobalDefaultCoordinateTolerance(14.21e-1)
        recon_volume = join.Execute([reconstructed['features'][k] for k in range(len(reconstructed['features']))])
        recon_volume.SetSpacing((float(md[0]), float(md[1]), float(md[2])))
        recon_volume.SetOrigin((float(md[3]), float(md[4]), float(md[5])))
        NIFTISingleSampleWriter(recon_volume, ID, new_folder_path)
        if train:
            recon_labels = join.Execute([reconstructed['labels'][k] for k in range(len(reconstructed['labels']))])
            recon_labels.SetSpacing((float(md[0]), float(md[1]), float(md[2])))
            recon_labels.SetOrigin((float(md[3]), float(md[4]), float(md[5])))
            NIFTISampleWriter(recon_volume, recon_labels, ID, new_folder_path)
            
    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 
                                     '''Module for the reconstruction of the original shape of the femur. 
                                     
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
                        help='Path to the csv containing the bounding boxes')
    
    parser.add_argument('metadata_path', 
                        metavar='metadata_path',
                        type = str, 
                        help='Path to the csv where each row contains the metadata of the original volume. ([spacing x y z origin x y x]) ')
    
    args = parser.parse_args() 
    
    new_folder_path = Path(args.new_folder_path)
    if new_folder_path.exists():
        pass
    else:
        new_folder_path.mkdir(exist_ok=True) 
           
    
    basepath = Path(args.basepath)
    
    patients, data_paths, masks_paths, data_folders = PathExplorer(basepath)
    
    ID = [[elem] for elem in data_folders]
    
    img_sizes = []
    for i in range(len(data_folders)):
        bounding_box = []
        data, data_array= DataLoad(data_paths[i], masks_paths[i]) 
        del(data_array)
        img_sizes.append(data['features'][0].GetSize()[2])
        with open(Path('{}'.format(args.csv_path)), 'r', newline='', encoding='UTF8') as f:
            reader = csv.reader(f, delimiter=',')
            lines = []
            for row in reader:
                lines.append(row)
            
            if i == 0:
                a = lines[0:(img_sizes[i] )]
                bounding_box.append(a)
            else:
                bounding_box.append(lines[img_sizes[i-1]:(img_sizes[i-1]+img_sizes[i])]) #-1
                
        metadata_list = []
        img_metadata = []
        with open(Path('{}'.format(args.metadata_path)), 'r', newline='', encoding='UTF8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                metadata_list.append(row)
        img_metadata.append(metadata_list[i])
        
        ReconstructionExtMetadata(data, ID[i], bounding_box, new_folder_path, img_metadata, train = args.train)