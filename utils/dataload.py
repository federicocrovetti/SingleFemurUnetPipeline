#dataload

import SimpleITK as sitk
from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter


def DicomReader(path):
    """

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


def DataLoad(data_path, masks_path):
    """
    Parameters
    ----------
    data_path : list of WindowsPaths to the folders containing the images (DICOM Series, NIFTI, ...)
    masks_path : list of WindowsPaths to the folders containing the labels (DICOM Series, NIFTI, ...)
    
    Returns
    -------
    data_and_labels : a dict type object containing sitk.Image format images and segmentations
    data_and_labels_array : a dict type object containing numpy format images and segmentations
    
    
    Batch load of the images in the selected paths. The preferred data formats to read are DICOM of NIFTI for
    the features and NIFTI or DICOM or NRRD for the labels.
    """
    data_and_labels = {'features': [], 'labels':[]}
    data_and_labels_array = {'features': [], 'labels':[]}
    
    for path in data_path:
        item_list = []
        for item in path.iterdir():
            item_list.append(item.name)
        if any(".dcm" in i for i in item_list):
            image , reader_dicom_data = DicomReader(path)
            image_array = sitk.GetArrayFromImage(image)
            data_and_labels['features'].append(image)
            data_and_labels_array['features'].append(image_array)
            
        else:
            for item in path.iterdir():
                item = str(item)
                image, reader_data = NiftiReader(item)
                image_array = sitk.GetArrayFromImage(image)
                data_and_labels['features'].append(image)
                data_and_labels_array['features'].append(image_array)
            
            
    for path in masks_path:
        item_list = []
        for item in path.iterdir():
            item_list.append(item.name)
        if any(".dcm" in i for i in item_list):
            image , reader_dicom_masks = DicomReader(path)
            image_array = sitk.GetArrayFromImage(image)
            data_and_labels['labels'].append(image)
            data_and_labels_array['labels'].append(image_array)
        
                    
        elif any(".nrrd" in i for i in item_list):
            for item in path.iterdir():
                item = str(item)
                image, reader_masks = NrrdReader(item)
                image_array = sitk.GetArrayFromImage(image)
                data_and_labels['labels'].append(image)
                data_and_labels_array['labels'].append(image_array)                              
                    
        else:
            for item in path.iterdir():
                item = str(item)
                image, reader_masks = NiftiReader(item)
                image_array = sitk.GetArrayFromImage(image)
                data_and_labels['labels'].append(image)
                data_and_labels_array['labels'].append(image_array)

            
    return data_and_labels, data_and_labels_array


def NIFTISampleWriter(volume_image, volume_mask, ID, new_folder_path):
    """
    Parameters
    ----------
    volume_image : SimpleITK format image
    volume_mask : SimpleITK format image
    ID: string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    new_folder_path : the parent folder for the new data

    -------
    Writes a Nifti format image both for data and labels in the folder identified by 'new_folder_path'
    with the 'mod' prefix

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
        
    masks_subfolder = new_patient_folder / 'mod{}Segmentation'.format(ID)
    
    if masks_subfolder.exists():
        pass
    else:
        masks_subfolder.mkdir(exist_ok=True)
        
    new_data_path = data_subfolder / 'mod{}.nii'.format(ID)
    sitk.WriteImage(volume_image, '{}'.format(new_data_path))
    
    new_mask_path = masks_subfolder / 'mod{}.nii'.format(ID)
    sitk.WriteImage(volume_mask, '{}'.format(new_mask_path))        

    return


if __name__ == '__main__':
    
    patients = []
    data = []
    masks = []
    data_folders = []
    
    parser = argparse.ArgumentParser(description = 
                                     '''Module for the batch data loading and preprocessing. With optional argument --resample operates a reshaping (deprecated for new versions in favour of greylevel thresholding.
                                     Take as input a variety of combination of data in the form of Dicom, Nifti and Nrrd.
                                     Recomended structure for the directory containing the data:
                                     
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
                        type = str, 
                        help='Path to the working directory in which the data is contained')
    
    parser.add_argument('-dl',
                        '--dataloader', 
                        type = str,
                        help='Path to the new directory onto which the resampled images will be written')
    
    parser.add_argument('-res', 
                        '--resample', 
                        type = str,
                        help='Path to the new directory onto which the resampled images will be written')
    args = parser.parse_args()
    
    if args.basepath:
        basepath = Path(args.basepath) 
        

    
    
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
    
    
    if args.dataloader:
        dataset, dataset_array = DataLoad(data, masks)

