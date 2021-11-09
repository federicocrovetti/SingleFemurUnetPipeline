import SimpleITK as sitk
from pathlib import Path
import numpy as np
import tensorflow as tf
import time
import argparse
from argparse import RawTextHelpFormatter


def DicomReader(path):
    reader_dicom = sitk.ImageSeriesReader()
    reader_dicom.SetImageIO("GDCMImageIO")
    reader_dicom.MetaDataDictionaryArrayUpdateOn()
    reader_dicom.LoadPrivateTagsOn()
    dicom_names = reader_dicom.GetGDCMSeriesFileNames('{}'.format(path))
    reader_dicom.SetFileNames(dicom_names)
    image = reader_dicom.Execute()
    return image, reader_dicom

def NiftiReader(path):
    reader_nifti = sitk.ImageFileReader()
    reader_nifti.SetImageIO("NiftiImageIO")
    reader_nifti.SetFileName('{}'.format(path))
    reader_nifti.LoadPrivateTagsOn()
    reader_nifti.ReadImageInformation()
    image = reader_nifti.Execute()
    return image, reader_nifti

def NrrdReader(path):
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
    meta_data_features: a list of tuples containing the names and the values of the DICOM (or NIFTI) metadata
                        of the loaded DICOM (or NIFTI) images
    meta_data_labels : a list of tuples containing the names and the values of the NIFTI (or DICOM/NRRD) metadata
                        of the loaded NIFTI (or DICOM/NRRD) images
    
    
    Batch load of the images in the selected paths. The preferred data formats to read are DICOM of NIFTI for
    the features and NIFTI or DICOM or NRRD for the labels.
    """
    data_and_labels = {'features': [], 'labels':[]}
    data_and_labels_array = {'features': [], 'labels':[]}
    
    meta_data_features = []
    meta_data_labels = []
    for path in data_path:
        item_list = []
        for item in path.iterdir():
            item_list.append(item.name)
        if any(".dcm" in i for i in item_list):
            image, reader_dicom_data = DicomReader(path)
            image_array = sitk.GetArrayFromImage(image)
            data_and_labels['features'].append(image)
            data_and_labels_array['features'].append(image_array)
            for i in range(image.GetDepth()):
                for k in reader_dicom_data.GetMetaDataKeys(i):
                    v = reader_dicom_data.GetMetaData(i,k)
                    meta_data_features.append((k,v))

            
        else:
            for item in path.iterdir():
                item = str(item)
                image, reader_data = NiftiReader(item)
                image_array = sitk.GetArrayFromImage(image)
                data_and_labels['features'].append(image)
                data_and_labels_array['features'].append(image_array)
                for i in range(image.GetDepth()):
                    for k in reader_data.GetMetaDataKeys(i):
                        v = reader_data.GetMetaData(i,k)
                        meta_data_features.append((k,v))

            
    for path in masks_path:
        item_list = []
        for item in path.iterdir():
            item_list.append(item.name)
        if any(".dcm" in i for i in item_list):
            image, reader_dicom_masks = DicomReader(path)
            image_array = sitk.GetArrayFromImage(image)
            data_and_labels['labels'].append(image)
            data_and_labels_array['labels'].append(image_array)
            #metadata
            for i in range(image.GetDepth()):
                for k in reader_dicom_masks.GetMetaDataKeys(i):
                    v = reader_dicom_masks.GetMetaData(i,k)
                    meta_data_labels.append((k,v))
                    
        elif any(".nrrd" in i for i in item_list):
            for item in path.iterdir():
                item = str(item)
                image, reader_masks = NrrdReader(item)
                image_array = sitk.GetArrayFromImage(image)
                data_and_labels['labels'].append(image)
                data_and_labels_array['labels'].append(image_array)
                for k in reader_masks.GetMetaDataKeys():
                    v = reader_masks.GetMetaData(k)
                    meta_data_labels.append((k,v))
            
                    
                    
        else:
            for item in path.iterdir():
                item = str(item)
                image, reader_masks = NiftiReader(item)
                image_array = sitk.GetArrayFromImage(image)
                data_and_labels['labels'].append(image)
                data_and_labels_array['labels'].append(image_array)
                for k in reader_masks.GetMetaDataKeys():
                    v = reader_masks.GetMetaData(k)
                    meta_data_labels.append((k,v))
            
    return data_and_labels, data_and_labels_array, meta_data_features, meta_data_labels

#dataset, dataset_array, metadata_images, metadata_masks = DataLoad(data, masks)

def DICOMSampleWriter(volume_image, ID, new_folder_path):
    """
    Parameters
    ----------
    
    volume_image : SimpleITK format image
    ID : string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    new_folder_path : the parent folder for the new data
    
    -------
    Creates a folder with the prefix 'mod', and a subfolder for the features,
    and writes a DICOM series.
    

    """

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    ID = str(ID)
    new_patient_folder = new_folder_path / 'mod{}'.format(ID)
    if new_patient_folder.exists():
        pass
    else:
        new_patient_folder.mkdir(exist_ok=True)
    
    data_subfolder = new_patient_folder / 'mod{}Data'.format(ID)
    data_subfolder.mkdir(exist_ok=True)

    
    img_depth = volume_image.GetDepth()
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    
    for i in range(img_depth):
        file = data_subfolder / '00{}.dcm'.format(i)
        img_slice = volume_image[:,:,i]
        img_slice.SetMetaData("0020,0032", '\\'.join(map(str,volume_image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        img_slice.SetMetaData("0020,0013", str(i))
        img_slice.SetMetaData("0010,0010", "Patient^{}^".format(ID))
        img_slice.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time)
        writer.SetFileName('{}'.format(file))
        writer.Execute(img_slice)
    return 

def NIFTISampleWriter(volume_image, ID, new_folder_path, image_and_mask = None, *volume_mask):
    """
    

    Parameters
    ----------
    volume_image : SimpleITK format image
    ID: string representing the sample's ID (patient's folder in the case of the resampling of a
         pre-existing image or the numerical ID with which to identify the folder of a new patient)
    new_folder_path : the parent folder for the new data
    
    image_and_mask : if 0 it writes both data and masks in NIFTI format
                     if 1 it writes only the data in NIFTI format
                     if >1 it writes only the masks in NIFTI
    volume_mask : SimpleITK format image (optional)
    -------
    Creates a folder with the prefix 'mod', 2 subfolders for the data and the masks,
    and writes a NIFTI file

    """
    
    ID = str(ID)
    new_patient_folder = new_folder_path / 'mod{}'.format(ID)
    if new_patient_folder.exists():
        pass
    else:
        new_patient_folder.mkdir(exist_ok=True)
    
    if image_and_mask == 0:
        data_subfolder = new_patient_folder / 'mod{}Data'.format(ID)
        data_subfolder.mkdir(exist_ok=True)
        masks_subfolder = new_patient_folder / 'mod{}Segmentation'.format(ID)
        masks_subfolder.mkdir(exist_ok=True)
        file = data_subfolder / 'mod{}.nii'.format(ID)
        sitk.WriteImage(volume_image, '{}'.format(file))
        new_mask_path = masks_subfolder / 'mod{}.nii'.format(ID)
        sitk.WriteImage(volume_mask, '{}'.format(new_mask_path))        
            
    elif image_and_mask == 1:
        data_subfolder = new_patient_folder / 'mod{}Data'.format(ID)
        data_subfolder.mkdir(exist_ok=True)

        file = data_subfolder / 'mod{}.nii'.format(ID)
        sitk.WriteImage(volume_image, '{}'.format(file))
        
    else:    
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        data_subfolder = new_patient_folder / 'mod{}Segmentation'.format(ID)
        data_subfolder.mkdir(exist_ok=True)

        file = data_subfolder / 'mod{}.nii'.format(ID)
        #sitk.WriteImage(volume_image, '{}'.format(file))
        writer.SetFileName('{}'.format(file))
        writer.Execute(volume_image)

    return



def TensorflowData(dataset):
    """
    
    Parameters
    ----------
    dataset : dict type obyect containing images and masks in numpy array format

    -------
    Returns the data in the form of tensorflow tensors

    """
    
    for i in range(len(dataset['features'])):
        dataset['features'][i] = dataset['features'][i][np.newaxis, ..., np.newaxis]
        dataset['features'][i] = tf.convert_to_tensor(dataset['features'][i], dtype =tf.float32)
        
    for i in range(len(dataset['labels'])):
        dataset['labels'][i] = dataset['labels'][i][np.newaxis, ..., np.newaxis]
        dataset['labels'][i] = tf.convert_to_tensor(dataset['labels'][i], dtype =tf.float32)    
    
    return dataset

# =============================================================================
#     resize_data(data, masks, data_folders, (0.78, 0.78, 1.0))
#     dataset, dataset_array, metadata_images, metadata_masks = DataLoad(data, masks)
#     tfdataset = TensorflowData(dataset)
# 
# 
# =============================================================================

def RespaceAndResize(data_path, masks_path, data_folders, new_folder_path, new_spacings=None, new_sizes=None):
    """
    
    Parameters
    ----------
    data_path : Path to the folders containing all of the features
    masks_path : Path to the folders containing all of the labels
    data_folders : list with the ID of the folders containing both the features and the labels subfolders
    new_folder_path : Path to the folder in which the outcome will be placed

    -------
    Creates a new dataset with the respaced and/or resized dataset

    """
    
    
    dataset, dataset_array, metadata_images, metadata_masks = DataLoad(data_path, masks_path)
    
    for i in range(len(dataset['features'])):
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        ID = data_folders[i]
        image = dataset['features'][i]
        origin = image.GetOrigin()
        direction = image.GetDirection()
        size = image.GetSize()
        spacing = image.GetSpacing()
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputDirection(direction)
        
        if new_spacings is not None:
            new_size = [o_size * o_spacing / n_spacing for o_size, o_spacing, n_spacing in zip(size, spacing, new_spacings)]
            new_size = list(map(lambda x : int(x), new_size))
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(new_spacings)
            print(new_size)
        elif new_sizes is not None:
            new_spacing = [size[i] * spacing[i] / float(ns) for i, ns in enumerate(new_sizes)]
            resampler.SetSize(new_sizes)
            resampler.SetOutputSpacing(new_spacing)
            print(new_spacing)
        
        resampled_image = resampler.Execute(image)
        DICOMSampleWriter(resampled_image, ID, new_folder_path, metadata_images)
       
    for i in range(len(dataset['labels'])):
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        ID = data_folders[i]
        image = dataset['labels'][i]
        origin = image.GetOrigin()
        direction = image.GetDirection()
        size = image.GetSize()
        spacing = image.GetSpacing()
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputDirection(direction)
        
        if new_spacings is not None:
            new_size = [o_size * o_spacing / n_spacing for o_size, o_spacing, n_spacing in zip(size, spacing, new_spacings)]
            new_size = list(map(lambda x : int(x), new_size))
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(new_spacings)
        elif new_sizes is not None:
            new_spacing = [size[i] * spacing[i] / float(ns) for i, ns in enumerate(new_sizes)]
            resampler.SetSize(new_sizes)
            resampler.SetOutputSpacing(new_spacing)
        
        
        resampled_image = resampler.Execute(image)
        NIFTISampleWriter(resampled_image, ID, new_folder_path, image_and_mask = 2)

    return

#RespaceAndResize(data, masks, data_folders, new_folder_path, new_spacings=None, new_sizes=(280,280,280))

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
    parser.add_argument('basepath', type = str, help='Path to the working directory in which the data is contained')
    parser.add_argument('-dl','--dataloader', type = str, #action = 'store_true',
                        help='Path to the new directory onto which the resampled images will be written')
    parser.add_argument('-res', '--resample', type = str, #action = 'store_true',
                        help='Path to the new directory onto which the resampled images will be written')
    args = parser.parse_args()
    basepath = Path(args.basepath) 
    
    
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__':
                print(item.name)
                data_folders.append(item.name)
                path = basepath / '{}'.format(item.name)
                patients.append(path)
                
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        if item.is_dir():
            if not item.name == '__pycache__':
                for elem in item.iterdir():
                    if "Data" in elem.name:
                        data.append(elem)
                    elif "Segmentation" in elem.name:
                        masks.append(elem)
        
    if args.dataloader:
       dataset, dataset_array, metadata_images, metadata_masks = DataLoad(data, masks)
    if args.resample:
       RespaceAndResize(data, masks, data_folders, Path(args.resample), new_spacings=None, new_sizes=(280,280,280))
                                          
                                        
                                        
