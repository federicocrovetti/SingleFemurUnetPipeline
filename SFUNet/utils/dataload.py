#dataload

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter


def DicomReader(path):
    """
    Reader for Dicom images
    Parameters
    ----------
    path :Pathlib path to the file

    Returns
    -------
    image : SimpleITK image
    reader_dicom : SimpleITK reader

    """
    reader_dicom = sitk.ImageSeriesReader()
    reader_dicom.SetImageIO("GDCMImageIO")
    reader_dicom.MetaDataDictionaryArrayUpdateOn()
    reader_dicom.LoadPrivateTagsOn()
    seriesID = reader_dicom.GetGDCMSeriesIDs('{}'.format(path))
    series_file_names = reader_dicom.GetGDCMSeriesFileNames('{}'.format(path), seriesID[0])
    reader_dicom.SetFileNames(series_file_names)
    image = reader_dicom.Execute()
    return image, reader_dicom


def NiftiReader(path):
    """
    Reader for Nifti images
    Parameters
    ----------
    path :Pathlib path to the file

    Returns
    -------
    image : SimpleITK image
    reader_dicom : SimpleITK reader

    """
    reader_nifti = sitk.ImageFileReader()
    reader_nifti.SetImageIO("NiftiImageIO")
    reader_nifti.SetFileName('{}'.format(path))
    reader_nifti.LoadPrivateTagsOn()
    reader_nifti.ReadImageInformation()
    image = reader_nifti.Execute()
    return image, reader_nifti


def NrrdReader(path):
    """
    Reader for Nrrd images
    Parameters
    ----------
    path :Pathlib path to the file

    Returns
    -------
    image : SimpleITK image
    reader_dicom : SimpleITK reader

    """
    reader_nrrd = sitk.ImageFileReader()
    reader_nrrd.SetImageIO("NrrdImageIO")
    reader_nrrd.SetFileName('{}'.format(path))
    reader_nrrd.LoadPrivateTagsOn()
    reader_nrrd.ReadImageInformation()
    image = reader_nrrd.Execute()
    return image, reader_nrrd


def PathExplorer(basepath):
    """
    This function reads the content of the parent folder and extract the paths to patients' features and labels,
    patient names and subfolder names.
    
    Parameters
    ----------
    basepath : Pathlib Path to the parent directory containing the data

    Returns
    -------
    patients : List of paths to the folders, containing data and masks, of each sample in the parent directory
    data : List of paths to all of the images in each 'patients' folder
    masks : List of paths to all of the masks in each 'patients' folder
    data_folders : List of names of the folders in parent directory

    """
    patients = []
    data = []
    masks = []
    data_folders = []
    
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__' and not item.name == '.hypothesis' and not item.name == '_logdir_' and not item.name == 'Fedz' and not item.name == '.pytest_cache':
                print(item.name)
                data_folders.append(item.name)
                path = basepath / '{}'.format(item.name)
                patients.append(path)
                for elem in item.iterdir():
                    if "Data" in elem.name:
                        data.append(elem)
                    elif "Segmentation" in elem.name:
                        masks.append(elem)
                
    return patients, data, masks, data_folders 

def DataLoad(data_path, masks_path):
    """
    This function loads in memory, under sitk.Image and numpy arrays, the whole content of the parent folder 
    by the list of paths directing to features and labels contained in the input lists.
    Can be used for both batch and sequential loading. The preferred data formats to read are DICOM of NIFTI for
    the features and NIFTI or DICOM or NRRD for the labels.
    
    Parameters
    ----------
    data_path : list of Pathlib Paths to the folders containing the images (DICOM Series, NIFTI, ...)
    masks_path : list of Pathlib Paths to the folders containing the labels (DICOM Series, NIFTI, ...)
    
    Returns
    -------
    data_and_labels : list of dict type object containing sitk.Image format images and segmentations
    data_and_labels_array : list of dict type object containing numpy format images and segmentations
    
    """
    
    data_and_labels = {'features': [], 'labels':[]}
    data_and_labels_array = {'features': [], 'labels':[]}
    
    item_list = []
    for item in data_path.iterdir():
        item_list.append(item.name)
    if any(".dcm" in i for i in item_list):
        image , reader_dicom_data = DicomReader(data_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, axes=[1, 2, 0])
        data_and_labels['features'].append(image)
        data_and_labels_array['features'].append(image_array)
        
    else:
        for item in data_path.iterdir():
            item = str(item)
            image, reader_data = NiftiReader(item)
            image_array = sitk.GetArrayFromImage(image)
            image_array = np.transpose(image_array, axes=[1, 2, 0])
            data_and_labels['features'].append(image)
            data_and_labels_array['features'].append(image_array)
   
    item_list = []
    for item in masks_path.iterdir():
        item_list.append(item.name)
    if any(".dcm" in i for i in item_list):
        image , reader_dicom_masks = DicomReader(masks_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.transpose(image_array, axes=[1, 2, 0])
        data_and_labels['labels'].append(image)
        data_and_labels_array['labels'].append(image_array)
    
                
    elif any(".nrrd" in i for i in item_list):
        for item in masks_path.iterdir():
            item = str(item)
            image, reader_masks = NrrdReader(item)
            image_array = sitk.GetArrayFromImage(image)
            image_array = np.transpose(image_array, axes=[1, 2, 0])
            data_and_labels['labels'].append(image)
            data_and_labels_array['labels'].append(image_array) 
            
    else:
        for item in masks_path.iterdir():
            item = str(item)
            image, reader_masks = NiftiReader(item)
            image_array = sitk.GetArrayFromImage(image)
            image_array = np.transpose(image_array, axes=[1, 2, 0])
            data_and_labels['labels'].append(image)
            data_and_labels_array['labels'].append(image_array)
            
    return data_and_labels, data_and_labels_array


def NIFTISampleWriter(volume_image, volume_mask, ID, new_folder_path):
    """
    Writes as a Nifti images the input sitk.Image(s), both features and labels, into the designed 
    parent folder with a folder structure equivalent to the required one which can be found on the readme file.
    The newly created subfolders and files will contain the prefix 'mod'.
    
    Parameters
    ----------
    volume_image : SimpleITK format image
    volume_mask : SimpleITK format image
    ID: string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    new_folder_path : the parent folder for the new data

    """
    
    ID = str(ID)
    new_patient_folder = new_folder_path / 'mod{}'.format(ID)
    if new_patient_folder.exists():
        pass
    else:
        new_patient_folder.mkdir(parents=True, exist_ok=True)
    
    data_subfolder = new_patient_folder / 'mod{}Data'.format(ID)
    
    if data_subfolder.exists():
        pass
    else:
        data_subfolder.mkdir(parents=True, exist_ok=True)
        
    masks_subfolder = new_patient_folder / 'mod{}Segmentation'.format(ID)
    
    if masks_subfolder.exists():
        pass
    else:
        masks_subfolder.mkdir(parents=True, exist_ok=True)
        
    new_data_path = data_subfolder / 'mod{}.nii'.format(ID)
    sitk.WriteImage(volume_image, '{}'.format(new_data_path))
    
    new_mask_path = masks_subfolder / 'mod{}.nii'.format(ID)
    sitk.WriteImage(volume_mask, '{}'.format(new_mask_path))        

    return

def NIFTISingleSampleWriter(volume_image, ID, new_folder_path):
    """
    Writes as a Nifti image the input sitk.Image into the designed 
    parent folder with a folder structure equivalent to the required one which can be found on the readme file.
    The newly created subfolders and files will contain the prefix 'mod'.
    
    Parameters
    ----------
    volume_image : SimpleITK format image
    ID: string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    new_folder_path : the parent folder for the new data

    """
    
    ID = str(ID)
    new_patient_folder = new_folder_path / 'mod{}'.format(ID)
    if new_patient_folder.exists():
        pass
    else:
        new_patient_folder.mkdir(exist_ok=True)
    
    data_subfolder = new_patient_folder / 'mod{}Data'.format(ID)
    
    if data_subfolder.exists():
        pass
    else:
        data_subfolder.mkdir(exist_ok=True)

        
    new_data_path = data_subfolder / 'mod{}.nii'.format(ID)
    sitk.WriteImage(volume_image, '{}'.format(new_data_path))      

    return

def MDTransfer(in_image, out_image):
    """
    Function for overwriting sitk.Image(s)' origin, direction and spacing inherited by
    another image.
    
    Parameters
    ----------
    in_image : SimpleITK image from which origin, direction and spacing will be copied
    out_image : SimpleITK image onto which origin, direction and spacing will be pasted

    """
    origin = in_image.GetOrigin()
    direction = in_image.GetDirection()
    spacing = in_image.GetSpacing()
    
    out_image.SetOrigin(origin)
    out_image.SetDirection(direction)
    out_image.SetSpacing(spacing)
    
    return


if __name__ == '__main__':
    pass
