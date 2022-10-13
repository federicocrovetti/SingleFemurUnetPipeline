import SimpleITK as sitk
from pathlib import Path
import numpy as np
import pytest
import hypothesis as hp
from hypothesis import given, settings
import hypothesis.strategies as st
from SFUNet.utils.dataload import NiftiReader, NIFTISampleWriter
from SFUNet.utils.slicer_utils import PathExplorerSlicedDataset, NIFTISlicesWriter, DatasetSlicerReorganizer, SliceDatasetLoader    

#Nomenclature for the redistribution of the sliced data
sets = ['Train', 'Val', 'Test']

legitimate_chars = st.characters(whitelist_categories=('Lu','Ll')
                                    ,min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=5,
                            max_size=30)

@st.composite
def slice_image_generator(draw):
  """
  Strategy for the generation of a single 2D slice
  """
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 2))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 2))
    direction = tuple([1., 0., 0., 1.])
    
    features = np.ones((25, 25))
    image = sitk.GetImageFromArray(features)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    
    return image

@st.composite
def image_generator(draw):
  """
  Strategy for the generation of a 3D image
  """
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([1., 0., 0., 0., 1., 0., 0., 0., 1.])
    
    features = np.ones((25, 25, 20))
    image = sitk.GetImageFromArray(features)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    
    return image

@settings(suppress_health_check=[hp.HealthCheck.too_slow], max_examples=2, deadline = None)
@given(image_generator(), image_generator(), text_strategy, text_strategy, st.sampled_from(sets))#st.lists(sets, min_size=1,max_size=1)) 
def test_NIFTISlicesWriter(datas, labelss, IDs, new_folder_paths, destinations):
  """
  Starting fro a 3D input image, we test the capability of NIFTISlicesWriter to write the input image,
  slice by slice, in the newly created folders identified by the prefix sampled from 'sets'.
  """
    data = datas
    labels = labelss
    ID = IDs
    destination = destinations
    
    basepath = Path.cwd()
    new_folder_path = basepath
    NIFTISlicesWriter(data, labels, ID, new_folder_path, destination)
    image = NiftiReader(new_folder_path / '{}Features'.format(destination)/ 'mod{}slice0.nii'.format(ID))
    image = image[0]
    mask = NiftiReader(new_folder_path / '{}Labels'.format(destination)/ 'mod{}slice0.nii'.format(ID))
    mask = mask[0]
    
    assert (sitk.GetArrayFromImage(image) == sitk.GetArrayFromImage(data)[0,:, :]).all()
    assert (sitk.GetArrayFromImage(mask) == sitk.GetArrayFromImage(labels)[0,:, :]).all()
    assert (image.GetDirection() == pytest.approx(data[0,:, :].GetDirection()))
    assert (mask.GetDirection() == pytest.approx(labels[0,:, :].GetDirection()))
    assert (image.GetSpacing() == pytest.approx(data[0,:, :].GetSpacing()))
    assert (mask.GetSpacing() == pytest.approx(labels[0,:, :].GetSpacing()))
    assert (image.GetPixelIDValue() == data[0,:, :].GetPixelIDValue())
    assert (mask.GetPixelIDValue() == labels[0,:, :].GetPixelIDValue())


@settings(suppress_health_check=[hp.HealthCheck.too_slow, hp.HealthCheck.filter_too_much], max_examples=10, deadline = None)   
@given(image_generator(), image_generator(), text_strategy, text_strategy, st.sampled_from(sets))
def test_DatasetSlicerReorganizer(datas, labelss, IDs, new_folder_paths, destinations):
  """
  Testing that DatasetSlicerReorganizer, starting from a parent folder with a single sample, 
  is capable of writing a new parent folder, with the structure given in the Readme, with 
  the original image sliced and contained in the proper Train-Val-Test folder.
  """
    
    data = datas
    labels = labelss
    ID = IDs
    
    train_samp = 1
    val_samp = 0
    test_samp = 0
    
    basepath = Path.cwd()
    new_path = Path('{}reorganizer'.format(new_folder_paths))
    new_folder_path = basepath / new_path
    print(new_folder_path)
    datalen = data.GetSize()[2]
    NIFTISampleWriter(data, labels, ID, new_folder_path)
    
    newfolderpath = new_folder_path / Path('{}reorganized'.format(new_folder_path))
    DatasetSlicerReorganizer(new_folder_path, newfolderpath, train_samp, val_samp, test_samp)
    trainfeat, trainmasks, valfeat, valmasks, testfeat, testmasks = PathExplorerSlicedDataset(newfolderpath)
    
    if train_samp != 0:
        for i in range(len(trainfeat)):
            img = NiftiReader(trainfeat[i])[0]
            assert (img.GetSize() == data[:,:,0].GetSize())
        assert (len(trainfeat) == datalen)
        assert (len(trainmasks) == datalen)
    elif val_samp != 0:
        for i in range(len(valfeat)):
            img = NiftiReader(valfeat[i])[0]
            assert (img.GetSize() == data[:,:,0].GetSize())
        assert (len(valfeat) == datalen)
        assert (len(valmasks) == datalen)
    elif test_samp != 0:
        for i in range(len(testfeat)):
            img = NiftiReader(testfeat[i])[0]
            assert (img.GetSize() == data[:,:,0].GetSize())
        assert (len(testfeat) == datalen)
        assert (len(testmasks) == datalen)
    
@settings(suppress_health_check=[hp.HealthCheck.too_slow], max_examples=2, deadline = None)
@given(image_generator(), image_generator(), st.integers(), st.integers(), st.integers(), text_strategy, text_strategy)
                           
def test_SliceDatasetLoader(datas, labelss, train_samps, val_samps, test_samps, IDs, new_folder_paths):
  """
  Testing that the 2D images loaded sequentially by SliceDatasetLoader correspond to the appropriate
  section (along x and y) of the original 3D image.
  """
    
    data = datas
    labels = labelss
    ID = IDs
    train_samp = train_samps
    val_samp = val_samps
    test_samp = test_samps
    
    basepath = Path.cwd()
    new_path = Path('{}'.format(new_folder_paths))
    new_folder_path = basepath / new_path
    print(new_folder_path)
    NIFTISampleWriter(data, labels, ID, new_folder_path)
    #reorganization
    newfolderpath = basepath
    DatasetSlicerReorganizer(new_folder_path, newfolderpath, train_samp, val_samp, test_samp)
    trainfeat, trainmasks, valfeat, valmasks, testfeat, testmasks = PathExplorerSlicedDataset(newfolderpath)

    if train_samp != 0:
        for i in range(len(trainfeat)):
            data, data_array = SliceDatasetLoader(trainfeat[i], trainmasks[i])
            assert (data['features'].GetSize() == data.GetSize()[0,1])
            assert (data_array['features'] == sitk.GetArrayFromImage(data)[1,2]).all()
            assert (data_array['labels'] == sitk.GetArrayFromImage(labels)[1,2]).all()
    elif val_samp != 0:
        for i in range(len(valfeat)):
            data, data_array = SliceDatasetLoader(valfeat[i], valmasks[i])
            assert (data['features'].GetSize() == data.GetSize()[0,1])
            assert (data_array['features'] == sitk.GetArrayFromImage(data)[1,2]).all()
            assert (data_array['labels'] == sitk.GetArrayFromImage(labels)[1,2]).all()
    elif test_samp != 0:
        for i in range(len(testfeat)):
            data, data_array = SliceDatasetLoader(testfeat[i], testmasks[i])
            assert (data['features'].GetSize() == data.GetSize()[0,1])
            assert (data_array['features'] == sitk.GetArrayFromImage(data)[1,2]).all()
            assert (data_array['labels'] == sitk.GetArrayFromImage(labels)[1,2]).all()
    
    
