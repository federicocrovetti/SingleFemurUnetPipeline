import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
from SFUNet.utils.dataload import PathExplorer, DicomReader, NiftiReader, NrrdReader


def PathExplorerSlicedDataset(basepath):
    """
    

    Parameters
    ----------
    basepath : Pathlib Path to the directory containing the data

    Returns
    -------
    trainfeat : List of Pathlib Paths to the images for training
    trainmasks : List of Pathlib Path to the labels for training
    valfeat : List of Pathlib Paths to the images for validation
    valmasks : List of Pathlib Path to the labels for validation
    testfeat : List of Pathlib Paths to the images for testing
    testmasks : List of Pathlib Path to the labels for testing
    
    N.B : it reliess on the specific naming scheme used across the package

    """
    trainfeat = []
    trainmasks = []
    valfeat = []
    valmasks = []
    testfeat = []
    testmasks = []
    folders = []
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        folders.append(item.name)
        if item.is_dir():
            if item.name == 'TrainFeatures':
                for elem in item.iterdir():
                    trainfeat.append(elem)
            elif item.name == 'TrainLabels':
                for elem in item.iterdir():
                    trainmasks.append(elem)
            elif item.name == 'ValFeatures':
                for elem in item.iterdir():
                    valfeat.append(elem)
            elif item.name == 'ValLabels':
                for elem in item.iterdir():
                    valmasks.append(elem)
            elif item.name == 'TestFeatures':
                for elem in item.iterdir():
                    testfeat.append(elem)
            elif item.name == 'TestLabels':
                for elem in item.iterdir():
                    testmasks.append(elem)
    
    return trainfeat, trainmasks, valfeat, valmasks, testfeat, testmasks



def NIFTISlicesWriter(volume_image, volume_mask, ID, new_folder_path, destination):#destination = None):
    """
    Parameters
    ----------
    volume_image : SimpleITK format image
    volume_mask : SimpleITK format image
    ID: string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    new_folder_path : the parent folder for the new data
    destination : String which identifies the rewritten image as train, validation or test sample. Is mandatory for the correct 
                  execution of the script for it to be choosen from the three accepted ones: Train, Val, Test
    -------
    Writes the input image and mask, decomposed into a single Nifti image per slice into one of three folders for images (Train-Val-Test:Features)
    and, correspondingly for the Labels, into one out of Train-Val-Test:Labels 
    """
    
    destination = str(destination)
    ID = str(ID)
    train_features_folder = new_folder_path / 'TrainFeatures'
    if train_features_folder.exists():
        pass
    else:
        train_features_folder.mkdir(parents=True, exist_ok=True)
    
    train_labels_folder = new_folder_path / 'TrainLabels'
    
    if train_labels_folder.exists():
        pass
    else:
        train_labels_folder.mkdir(parents=True, exist_ok=True)
    
    val_features_folder = new_folder_path / 'ValFeatures'
    if val_features_folder.exists():
        pass
    else:
        val_features_folder.mkdir(parents=True, exist_ok=True)
    
    val_labels_folder = new_folder_path / 'ValLabels'
    
    if val_labels_folder.exists():
        pass
    else:
        val_labels_folder.mkdir(parents=True, exist_ok=True)
    test_features_folder = new_folder_path / 'TestFeatures'
    if test_features_folder.exists():
        pass
    else:
        test_features_folder.mkdir(parents=True, exist_ok=True)
    
    test_labels_folder = new_folder_path / 'TestLabels'
    
    if test_labels_folder.exists():
        pass
    else:
        test_labels_folder.mkdir(parents=True, exist_ok=True)
        
    if 'Train' in destination:
        for i in range(volume_image.GetSize()[2]):
            new_data_path = train_features_folder / 'mod{}slice{}.nii'.format(ID,i)
            sitk.WriteImage(volume_image[:,:,i], '{}'.format(new_data_path))
        
        for i in range(volume_mask.GetSize()[2]):
            new_data_path = train_labels_folder / 'mod{}slice{}.nii'.format(ID,i)
            sitk.WriteImage(volume_mask[:,:,i], '{}'.format(new_data_path))
            
    elif 'Val' in destination:
        for i in range(volume_image.GetSize()[2]):
            new_data_path = val_features_folder / 'mod{}slice{}.nii'.format(ID,i)
            sitk.WriteImage(volume_image[:,:,i], '{}'.format(new_data_path))
        
        for i in range(volume_mask.GetSize()[2]):
            new_data_path = val_labels_folder / 'mod{}slice{}.nii'.format(ID,i)
            sitk.WriteImage(volume_mask[:,:,i], '{}'.format(new_data_path))
            
    elif 'Test' in destination:
        for i in range(volume_image.GetSize()[2]):
            new_data_path = test_features_folder / 'mod{}slice{}.nii'.format(ID,i)
            sitk.WriteImage(volume_image[:,:,i], '{}'.format(new_data_path))
        
        for i in range(volume_mask.GetSize()[2]):
            new_data_path = test_labels_folder / 'mod{}slice{}.nii'.format(ID,i)
            sitk.WriteImage(volume_mask[:,:,i], '{}'.format(new_data_path))
            
    else:
        raise ValueError("{} was designed as destination, instead of the accepted ones".format(destination))

    return

#Riorganizzatore che riscrive le immagini in slices singole, ma con la classica struttura a subfolders 
#e non quella di train,val,test.

def DatasetSlicerReorganizer(basepath, newfolderpath, train_samp, val_samp, test_samp):
    """
    

    Parameters
    ----------
    basepath : Pathlib Path to the directory containing the data
    newfolderpath : the parent folder for the new data
    train_samp : integer; The number of images intended to be used as training set out of total samples available 
    val_samp : integer; The number of images intended to be used as validation set out of total samples available 
    test_samp : integer; The number of images intended to be used as testing set out of total samples available 
    
    -------
    Scans the parent folder containing the data and rewrites it as a sequence of single Nifti images, divided accordingly into 
    Train, Val and Test sets' folders

    """
    
    patients, image_paths, masks_paths, data = PathExplorer(basepath)
    print(image_paths)
    masks = []
    imgs = []
    imgs_names = []
    masks_names = []
    for j in range(len(image_paths)):
        for item in image_paths[j].iterdir():
            imgs.append(item)
    for j in range(len(masks_paths)):
        for item in masks_paths[j].iterdir():
            masks.append(item)
    for k in range(len(image_paths)):
        for item in image_paths[k].iterdir():
            imgs_names.append(item.name)
    for l in range(len(masks_paths)):
        for item in masks_paths[l].iterdir():
            masks_names.append(item.name)
   
    for i in range(0, train_samp):
        if any(".dcm" in elem for elem in imgs_names):
            img = DicomReader(image_paths[i])[0]
        if any(".nii" in elem for elem in imgs_names):
            img = NiftiReader(imgs[i])[0]
        if any(".nrrd" in elem for elem in imgs_names):
            img = NrrdReader(imgs[i])[0]
        if any(".dcm" in elem for elem in masks_names):
            mask = DicomReader(masks_paths[i])[0]
        if any(".nii" in elem for elem in masks_names):
            mask = NiftiReader(str(masks[i]))[0]
        if any(".nrrd" in elem for elem in masks_names):
            mask = NrrdReader(str(masks[i]))[0]
        NIFTISlicesWriter(img, mask, data[i], newfolderpath, destination = 'Train')
        
    for i in range(train_samp, train_samp + val_samp):
        if any(".dcm" in elem for elem in imgs_names):
            img = DicomReader(image_paths[i])[0]
        if any(".nii" in elem for elem in imgs_names):
            img = NiftiReader(imgs[i])[0]
        if any(".nrrd" in elem for elem in imgs_names):
            img = NrrdReader(imgs[i])[0]
        if any(".dcm" in elem for elem in masks_names):
            mask = DicomReader(masks_paths[i])[0]
        if any(".nii" in elem for elem in masks_names):
            mask = NiftiReader(str(masks[i]))[0]
        if any(".nrrd" in elem for elem in masks_names):
            mask = NrrdReader(str(masks[i]))[0]
        NIFTISlicesWriter(img, mask, data[i], newfolderpath, destination = 'Val')
        
    if test_samp != 0:
        for i in range(train_samp + val_samp, train_samp + val_samp + test_samp):
            if any(".dcm" in elem for elem in imgs_names):
                img = DicomReader(image_paths[i])[0]
            if any(".nii" in elem for elem in imgs_names):
                img = NiftiReader(imgs[i])[0]
            if any(".nrrd" in elem for elem in imgs_names):
                img = NrrdReader(imgs[i])[0]
            if any(".dcm" in elem for elem in masks_names):
                mask = DicomReader(masks_paths[i])[0]
            if any(".nii" in elem for elem in masks_names):
                mask = NiftiReader(str(masks[i]))[0]
            if any(".nrrd" in elem for elem in masks_names):
                mask = NrrdReader(str(masks[i]))[0]
            NIFTISlicesWriter(img, mask, data[i], newfolderpath, destination = 'Test')
    else:
        pass
    return

#%%

def SliceDatasetLoader(data_path, masks_path):
    """
    

    Parameters
    ----------
    data_path : List containing the paths to images
    masks_path : List containing the paths to masks

    Returns
    -------
    data_and_labels : List of Dict type objecst, with keys Features and Labels, containing sitk images
    data_and_labels_array : List of Dict type objects, with keys Features and Labels, containing numpy arrays

    """
    #data_path è il path fino a train/val/test
    #just a single patient at a time. Basta che nella lista ci sia un singolo path
    data_and_labels = {'features': [], 'labels':[]}
    data_and_labels_array = {'features': [], 'labels':[]}
    
    dataname = str(data_path)
    
    if ".dcm" in dataname:
        image , reader_dicom_data = DicomReader(data_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, axes=[1, 0])
        data_and_labels['features'].append(image)
        data_and_labels_array['features'].append(image_array)
        
    elif ".nii" in dataname:
        image, reader_data = NiftiReader(data_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, axes=[1, 0])
        data_and_labels['features'].append(image)
        data_and_labels_array['features'].append(image_array)
    else:
        image, reader_data = NrrdReader(data_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, axes=[1, 0])
        data_and_labels['features'].append(image)
        data_and_labels_array['features'].append(image_array)
       
    labelname = str(masks_path)
    
    if ".dcm" in labelname:
        image , reader_dicom_masks = DicomReader(masks_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, axes=[1, 0])
        data_and_labels['labels'].append(image)
        data_and_labels_array['labels'].append(image_array)
    
                
    elif ".nrrd" in labelname:
        image, reader_masks = NrrdReader(masks_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, axes=[1, 0])
        data_and_labels['labels'].append(image)
        data_and_labels_array['labels'].append(image_array) 
            
    else:
        image, reader_masks = NiftiReader(masks_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, axes=[1, 0])
        data_and_labels['labels'].append(image)
        data_and_labels_array['labels'].append(image_array)
    
    return data_and_labels, data_and_labels_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 
                                     '''M
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
    
    parser.add_argument('train_perc',
                        metavar = 'train_perc',
                        type = int,
                        help = 'Number of patients for training')
    
    parser.add_argument('val_perc',
                        metavar = 'val_perc',
                        type = int,
                        help = 'Number of patients for validation')
    
    parser.add_argument('test_perc',
                        metavar = 'test_perc',
                        type = int,
                        help = 'Number of patients for testing')
    
    
    args = parser.parse_args() 
    
    new_folder_path = Path(args.new_folder_path)
    if new_folder_path.exists():
        pass
    else:
        new_folder_path.mkdir(exist_ok=True) 
    
    basepath = Path(args.basepath)
    new_folder_path = Path(args.new_folder_path)
   
    DatasetSlicerReorganizer(basepath, new_folder_path, args.train_perc, args.val_perc, args.test_perc)