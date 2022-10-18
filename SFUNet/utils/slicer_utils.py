import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
from SFUNet.utils.dataload import PathExplorer, DicomReader, NiftiReader, NrrdReader


def PathExplorerSlicedDataset(basepath):
    """
    This function reads the content of the parent folder and extract the paths to images to be used in
    Training, Validation or Testing.
    It's supposed to be used on the newly reorganized dataset.

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
    Writes as a series of Nifti images (one per slice) the input sitk.Image(s), both features and labels, into the designed 
    parent folder with a folder structure composed by six folders: Train/Val/Test-Features/Labels
    The newly created files will contain the prefix 'mod'.
    
    Parameters
    ----------
    volume_image : SimpleITK format image
    volume_mask : SimpleITK format image
    ID: string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    new_folder_path : the parent folder for the new data
    destination : String which identifies the rewritten image as train, validation or test sample. Is mandatory for the correct 
                  execution of the script for it to be choosen from the three accepted ones: Train, Val, Test
    
    """
    
    ID = str(ID)
    
    if destination:
        destination = str(destination)
        new_features_folder = new_folder_path / '{}Features'.format(destination)
        new_features_folder.mkdir(parents=True, exist_ok=True)
        
        new_labels_folder = new_folder_path / '{}Labels'.format(destination)
        new_labels_folder.mkdir(parents=True, exist_ok=True)
        
        for i in range(volume_image.GetSize()[2]):
            new_data_path = new_features_folder / 'mod{}slice{}.nii'.format(ID,i)
            sitk.WriteImage(volume_image[:,:,i], '{}'.format(new_data_path))
        
        for i in range(volume_mask.GetSize()[2]):
            new_data_path = new_labels_folder / 'mod{}slice{}.nii'.format(ID,i)
            sitk.WriteImage(volume_mask[:,:,i], '{}'.format(new_data_path))
            
    else:
        ValueError("destination wasn't assigned. Use one of the accepted ones: Train, Val, Test")

    return

#Riorganizzatore che riscrive le immagini in slices singole, ma con la classica struttura a subfolders 
#e non quella di train,val,test.

def DatasetSlicerReorganizer(basepath, newfolderpath, train_samp, val_samp, test_samp):
    """
    Scans the parent folder containing the data and rewrites it as a sequence of single Nifti images, divided accordingly into 
    Train, Val and Test sets' folders. This is done both for Features nd Labels

    Parameters
    ----------
    basepath : Pathlib Path to the directory containing the data
    newfolderpath : the parent folder for the new data
    train_samp : integer; The number of images intended to be used as training set out of total samples available 
    val_samp : integer; The number of images intended to be used as validation set out of total samples available 
    test_samp : integer; The number of images intended to be used as testing set out of total samples available 
    

    """
    
    patients, image_paths, masks_paths, data = PathExplorer(basepath)
    imgs = []
    imgs_names = []
    masks = []
    masks_names = []
    for j in range(len(image_paths)):
        for item in image_paths[j].iterdir():
            imgs.append(item)
            imgs_names.append(item.name)
            
        for item in masks_paths[j].iterdir():
            masks.append(item)
            masks_names.append(item.name)
    
    sets = [0, train_samp, val_samp, test_samp]
    location = {0 : 'Train', 1 : 'Val', 2 : 'Test'}
    execution = {'.dcm' : DicomReader, '.nii' : NiftiReader, 'nrrd' : NrrdReader}
    
    for i in range(len(sets)-1):
        for j in range(sets[i], sets[i+1]):
            for key in execution.keys():
                if key in imgs_names[j]:
                    if key == '.dcm':
                        img = execution[key](image_paths[j])[0]
                    else:
                        img = execution[key](imgs[j])[0]
                else:
                    Exception('{} is not of any supported file extension'.format(imgs_names[j]))
                
                if key in masks_names[j]:
                    if key == '.dcm':
                        mask = execution[key](masks_paths[j])[0]
                    else:
                        mask = execution[key](masks[j])[0]
                else:
                    Exception('{} is not of any supported file extension'.format(masks_names[j]))
            NIFTISlicesWriter(img, mask, data[i], newfolderpath, destination = location[i])
    return


def SliceDatasetLoader(data_path, masks_path):
    """
    This function loads in memory, under sitk.Image and numpy arrays, the whole content of the parent folder 
    by the list of paths directing to features and labels contained in the input lists.
    Can be used for both batch and sequential loading. The preferred data formats to read are DICOM of NIFTI for
    the features and NIFTI or DICOM or NRRD for the labels.

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
    execution = {'.dcm' : DicomReader, '.nii' : NiftiReader, 'nrrd' : NrrdReader}
    
    dataname = str(data_path)
    for key in execution.keys():
        if key in dataname:
            image , reader_dicom_data = execution[key](data_path)
            image_array = sitk.GetArrayFromImage(image)
            image_array = np.transpose(image_array, axes=[1, 0])
            data_and_labels['features'].append(image)
            data_and_labels_array['features'].append(image_array)
        else:
            Exception('{} is not of any supported file extension'.format(dataname))
            
    labelname = str(masks_path)
    for key in execution.keys():
        if key in labelname:
            image , reader_dicom_data = execution[key](masks_path)
            image_array = sitk.GetArrayFromImage(image)
            image_array = np.transpose(image_array, axes=[1, 0])
            data_and_labels['features'].append(image)
            data_and_labels_array['features'].append(image_array)
        else:
            Exception('{} is not of any supported file extension'.format(dataname))
    
    return data_and_labels, data_and_labels_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 
                                     '''Starting form a parent folder with the structure provided in the readme file,
                                     the script rewrites all of the data contained there divided into six folders:
                                         Train/Val/Test-Features/Labels
                                         Into each folder there will be the first upper third of each patient's image/label
                                         sliced into single nifti images, one for each slice in the original image.
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