from SFUNet.utils.bonereconstruction import ReconstructionExtMetadata
from SFUNet.utils.dataload import NiftiReader, SequentialLoader, PathExplorer
import SimpleITK as sitk 
import csv
import numpy as np
import pytest 
import hypothesis as hp
import hypothesis.strategies as st
from hypothesis import given, settings 
from pathlib import Path


legitimate_chars = st.characters(whitelist_categories=('Lu','Ll')
                                    ,min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=5,
                            max_size=30)


@st.composite
def skewed_image_generator(draw):
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([1., 0., 0., 0., 1., 0., 0., 0., 1.])
    
    bbox = []
    features = np.ones((40, 256, 256))
        #[BBOXX1, BBOXX2, BBOXY1, BBOXY2, YMIN, YMAX]
    for i in range(features.shape[0]):
        features[i, 90+i:120+i, 90+i:120+i] = 2
        bbox.append([0, 40, 90, 120, 0, 256])
    image = sitk.GetImageFromArray(features)
    image = sitk.Cast(image, sitk.sitkInt16)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    labels = sitk.GetImageFromArray(features)
    labels = sitk.Cast(labels, sitk.sitkInt16)
    labels.SetOrigin(origin)
    labels.SetSpacing(spacing)
    labels.SetDirection(direction)
    dataset = {'features' : [], 'labels' : []}
    dataset['features'].append(image)
    dataset['labels'].append(labels)
    metadata = [spacing[0], spacing[1], spacing[2], origin[0], origin[1], origin[2]]
    
    return dataset, bbox, metadata 

@settings(suppress_health_check=[hp.HealthCheck.too_slow], max_examples=4, deadline = None)
@given(skewed_image_generator(), text_strategy, text_strategy)
def test_ReconstructionExtMetadata(gen, IDs, new_folder_paths):
    data = gen[0]
    bbox = [gen[1]]
    metadata = gen[2]
    ID = IDs
    new_folder_path = new_folder_paths
    basepath = Path.cwd()
    new_path = Path('{}'.format(new_folder_paths))
    new_folder_path = basepath / new_path
    ReconstructionExtMetadata(data, ID, bbox, new_folder_path, metadata, train = True)
    
    patients, data_paths, masks_paths, data_folders = PathExplorer(new_folder_path)
    for i in range(len(data_paths)):
        dataset, dataset_array = SequentialLoader(data_paths[i], masks_paths[i])
        fac_simile = sitk.Image(256, 512, 40, sitk.sitkInt16)
        fac_simile[:,0:256,:] = 1
        fac_simile[90:120, 90:120, :] = 2
        spacing = dataset['features'][0].GetSpacing()
        origin = dataset['features'][0].GetOrigin()
        direction = dataset['features'][0].GetDirection()
        fac_simile.SetSpacing(spacing)
        fac_simile.SetOrigin(origin)
        fac_simile.SetDirection(direction)
        
        assert(dataset['features'][0] == fac_simile)
