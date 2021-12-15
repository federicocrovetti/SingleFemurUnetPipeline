# -*- coding: utf-8 -*-
#Test cropping.py

import SimpleITK as sitk
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as st
from dataload import NIFTISampleWriter
from dataload import NiftiReader
from cropping import Crop


#STRATEGIES

legitimate_chars = st.characters(whitelist_categories=('Lu','Ll')
                                    ,min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=5,
                            max_size=30)

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

bounds_strategy = st.lists(st.integers(), min_size=6, max_size=6)



#TESTS
@given(image_generator(), bounds_strategy, text_strategy, text_strategy)
def test_Crop(datasets, label_sizess, IDs, new_folder_paths):
    images = datasets
    dataset = {'features': [], 'labels':[]}
    dataset['features'].append(images)
    dataset['labels'].append(images)
    label_sizes = label_sizess
    ID = IDs
    new_folder_path = working_directory_path / Path('{}'.format(new_folder_paths))
    
    Crop(dataset, label_sizes, ID, new_folder_path)
    image = NiftiReader(new_folder_path / 'mod{}'.format(ID)/ 'mod{}Data'.format(ID)/ 'mod{}.nii'.format(ID))
    mask = NiftiReader(new_folder_path / 'mod{}'.format(ID)/ 'mod{}Segmentation'.format(ID)/ 'mod{}.nii'.format(ID))
    image = image[0]
    mask = mask[0]
    assert (datasets['features'][0].GetDirection() == pytest.approx(image.GetDirection()))
    assert (datasets['labels'][0].GetDirection() == pytest.approx(mask.GetDirection()))
    assert (datasets['features'][0].GetSpacing() == pytest.approx(image.GetSpacing()))
    assert (datasets['labels'][0].GetSpacing() == pytest.approx(mask.GetSpacing()))
    assert(datasets['features'][0].GetSize() == image.GetSize())
    assert(datasets['labels'][0].GetSize() == mask.GetSize())
































