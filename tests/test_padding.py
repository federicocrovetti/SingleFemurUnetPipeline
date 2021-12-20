# -*- coding: utf-8 -*-
#test padding.py

import SimpleITK as sitk
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as st
from dataload import NiftiReader
from padding import SquareComplete, Padding



#Strategies

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


size_strategy2D = st.lists(st.integers(min_value=0, max_value=40), min_size=2, max_size=2)


#Tests

@given(image_generator(), size_strategy2D)
def test_SquareComplete(images, req_size):
    dataset = images
    dataset = {'features': [], 'labels':[]}
    dataset['features'].append(images)
    dataset['labels'].append(images)
    req_sizes = req_size
    
    image_sizes = [dataset['features'][i].GetSize() for i in range(len(dataset['features']))]
    label_sizes = [dataset['labels'][i].GetSize() for i in range(len(dataset['labels']))]
    pad_sizes = SquareComplete(dataset, req_sizes)
    
    for i in range(len(image_sizes)):
        assert(image_sizes[i] + pad_sizes[i] == req_sizes)
    
    for i in range(len(label_sizes)):
        assert(label_sizes[i] + pad_sizes[i] == req_sizes)


@given(image_generator(), text_strategy, text_strategy, size_strategy2D, st.integers(min_value = -1000, max_value=3000))
def test_Padding(datasets, IDs, new_paths, up_bounds, constants):
    images = datasets
    dataset = {'features': [], 'labels':[]}
    dataset['features'].append(images)
    dataset['labels'].append(images)
    up_bound = up_bounds
    ID = IDs
    new_folder_path = working_directory_path / Path('{}'.format(new_folder_paths))
    constant = constants
    
    Padding(dataset, ID, new_folder_path, up_bound, constant = constant)
    image = NiftiReader(new_folder_path / 'mod{}'.format(ID)/ 'mod{}Data'.format(ID)/ 'mod{}.nii'.format(ID))
    mask = NiftiReader(new_folder_path / 'mod{}'.format(ID)/ 'mod{}Segmentation'.format(ID)/ 'mod{}.nii'.format(ID))
    image = image[0]
    mask = mask[0]
    
    assert((datasets['features'][0].GetSize()[0] + up_bounds[0]) == image.GetSize()[0])
    assert((datasets['features'][0].GetSize()[1] + up_bounds[1]) == image.GetSize()[1])
    assert((datasets['features'][0].GetSize()[2] + up_bounds[2]) == image.GetSize()[2])
    assert((datasets['labels'][0].GetSize()[0] + up_bounds[0]) == mask.GetSize()[0])
    assert((datasets['labels'][0].GetSize()[1] + up_bounds[1]) == mask.GetSize()[1])
    assert((datasets['labels'][0].GetSize()[2] + up_bounds[2]) == mask.GetSize()[2])









