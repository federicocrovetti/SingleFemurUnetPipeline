import numpy as np
import SimpleITK as sitk
from pathlib import Path
from _dataload_and_preprocessing_ import DICOMSampleWriter, NIFTISampleWriter

new_folder_path = Path(r'D:/Fedz/Pattern_Recognition_Project/HipOp_OK_resampled')

"""
Generates a folder with a combination of translation and rotation for the selected DICOM series and 
the corresponding NIFTI file containing the labels.
  
image : SimpleITK image of the CT scans
label : SimpleITK image of the labels
theta_z : angle of the rotation about z-axis 
ID : name of the patient's folder
   
"""

def rotation3d(image, label, theta_z, metadata, ID):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk MRI image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param show: Boolean, whether or not the user wants to see the result of the rotation
    :return: The rotated image
    """
    def matrix_from_axis_angle(a):
        """ Compute rotation matrix from axis-angle.
        This is called exponential map or Rodrigues' formula.
        Parameters
        ----------
        a : array-like, shape (4,)
            Axis of rotation and rotation angle: (x, y, z, angle)
        Returns
        -------
        R : array-like, shape (3, 3)
            Rotation matrix
        """
        ux, uy, uz, theta = a
        c = np.cos(theta)
        s = np.sin(theta)
        ci = 1.0 - c
        R = np.array([[ci * ux * ux + c,
                       ci * ux * uy - uz * s,
                       ci * ux * uz + uy * s],
                      [ci * uy * ux + uz * s,
                       ci * uy * uy + c,
                       ci * uy * uz - ux * s],
                      [ci * uz * ux - uy * s,
                       ci * uz * uy + ux * s,
                       ci * uz * uz + c],
                      ])
        
        return R
    
    def resample(image, transform, interpolator):
        """
        This function resamples (updates) an image using a specified transform
        :param image: The sitk image we are trying to transform
        :param transform: An sitk transform (ex. resizing, rotation, etc.
                                             :return: The transformed sitk image
                                             """
        reference_image = image
        #interpolator = sitk.sitkLinear
        default_value = 0
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value) 
    
    def get_center(img):
        """
        This function returns the physical center point of a 3d sitk image
        :param img: The sitk image we are trying to find the center of
        :return: The physical center point of the image
        """
        width, height, depth = img.GetSize()
        return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                                  int(np.ceil(height/2)),
                                                  int(np.ceil(depth/2))))

    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform()
    #print(euler_transform.GetMatrix())
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)

    direction = image.GetDirection()
    axis_angle = (direction[2], direction[5], direction[8], theta_z)
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())
    resampled_image = resample(image, euler_transform, sitk.sitkLinear)
    DICOMSampleWriter(resampled_image, ID, new_folder_path, metadata)
    
    #label
    direction = label.GetDirection()
    axis_angle = (direction[2], direction[5], direction[8], theta_z)
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())
    resampled_label = resample(label, euler_transform, sitk.sitkNearestNeighbor)
    NIFTISampleWriter(resampled_label , ID, new_folder_path, image_and_mask = 2)
    return 


def FlipDataAugmentation(image, label, ID, metadata_images_, metadata_masks_):
    """
    Generates a folder with a flipped transformation for the selected DICOM series and the corresponding
    NIFTI file containing the labels.
    The transformation matrix is fixed.
    
    image : SimpleITK image of the CT scans
    label : SimpleITK image of the labels
    ID : Random serial number for the suffix of the augmented data folder generated

    """
    
    dimension = image.GetDimension()
    reference_physical_size = np.zeros(dimension)
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(image.GetSize(), image.GetSpacing(), reference_physical_size)]
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()    
    reference_size = image.GetSize()
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ] 
    
    reference_image = sitk.Image(reference_size, image[0].GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
        
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(image.GetDirection())
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)
    centering_transform = sitk.TranslationTransform(dimension)
    image_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(image_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    comp_trans = sitk.CompositeTransform([centered_transform, centering_transform])
        
    flipped_transform = sitk.AffineTransform(dimension)
    flipped_transform.SetCenter(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    #matrix which defines the kind of flipping:
    flipped_transform.SetMatrix([1,0,0,0,1,0,0,0,-1])
    full_transform = sitk.CompositeTransform([comp_trans, flipped_transform])
    #image
    flipped_image = sitk.Resample(image, reference_image, full_transform, sitk.sitkLinear, 0.0)
    DICOMSampleWriter(flipped_image, metadata_images_, ID)
    #label
    flipped_label = sitk.Resample(label, reference_image, full_transform, sitk.sitkLinear, 0.0)
    NIFTISampleWriter(flipped_label, ID, image_and_mask = 2, metadata_images= metadata_images_, metadata_masks = metadata_masks_)
    
    return
