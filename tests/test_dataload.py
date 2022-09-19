import SimpleITK as sitk
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis as hp
import hypothesis.strategies as st
from SFUNet.utils.dataload import PathExplorer, DataLoad, NiftiReader, NIFTISampleWriter

legitimate_chars = st.characters(whitelist_categories=('Lu','Ll')
                                    ,min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=5,
                            max_size=30)


@st.composite
def slice_image_generator(draw):
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
@given(image_generator(), image_generator(), text_strategy, text_strategy) #text_strategy path_generator
def test_NIFTISampleWriter(datas, labelss, IDs, new_folder_paths): #new_folder_paths
    data = datas
    labels = labelss
    ID = IDs
    basepath = Path.cwd()
    new_path = Path('{}'.format(new_folder_paths))
    new_folder_path = basepath / new_path
    
    NIFTISampleWriter(data, labels, ID, new_folder_path)
    image = NiftiReader(new_folder_path / 'mod{}'.format(ID)/ 'mod{}Data'.format(ID)/ 'mod{}.nii'.format(ID))
    mask = NiftiReader(new_folder_path / 'mod{}'.format(ID)/ 'mod{}Segmentation'.format(ID)/ 'mod{}.nii'.format(ID))
    image = image[0]
    mask = mask[0]
    
    assert (sitk.GetArrayFromImage(image) == sitk.GetArrayFromImage(data)).all()
    assert (sitk.GetArrayFromImage(mask) == sitk.GetArrayFromImage(labels)).all()
    assert (image.GetDirection() == pytest.approx(data.GetDirection()))
    assert (mask.GetDirection() == pytest.approx(labels.GetDirection()))
    assert (image.GetSpacing() == pytest.approx(data.GetSpacing()))
    assert (mask.GetSpacing() == pytest.approx(labels.GetSpacing()))
    assert (image.GetPixelIDValue() == data.GetPixelIDValue())
    
@settings(suppress_health_check=[hp.HealthCheck.too_slow, hp.HealthCheck.filter_too_much], max_examples=2, deadline = None)   
@given(image_generator(), image_generator(), st.lists(st.text(alphabet=legitimate_chars, min_size=5,max_size = 8), min_size = 2, max_size = 2), text_strategy) #, st.integers(min_value = 2, max_value = 2)
def test_DataLoad(datas, labelss, IDs, new_folder_pathss): #, sampless
    data = datas
    labels = labelss
    samples = 2
    ID = IDs
    new_folder_paths = new_folder_pathss
    
    basepath = Path.cwd()
    new_path = Path('sequ{}'.format(new_folder_paths))
    new_folder_path = basepath / new_path
    
    for i in range(samples):
        NIFTISampleWriter(data, labels, ID[i], new_folder_path)
    patients, data_paths, masks_paths, data_folders = PathExplorer(new_folder_path)

    for i in range(len(data_paths)):
        dataset, dataset_array = DataLoad(data_paths[i], masks_paths[i])
        assert (dataset['features'][0].GetSize() == data.GetSize())  
        assert (dataset['labels'][0].GetSize() == labels.GetSize())  
        assert (dataset['features'][0].GetDirection() == pytest.approx(data.GetDirection()))
        assert (dataset['labels'][0].GetDirection() == pytest.approx(labels.GetDirection()))
 








