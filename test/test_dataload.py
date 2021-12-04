# -*- coding: utf-8 -*-

import SimpleITK as sitk
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from dataload import NIFTISampleWriter
from dataload import NiftiReader


#path/text generation
legitimate_chars = st.characters(whitelist_categories=('Lu','Ll')
                                    ,min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=5,
                            max_size=30)


#strategy for image generation

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

@given(image_generator(), image_generator(), text_strategy, text_strategy) 
def test_NIFTISampleWriter(datas, labelss, IDs, new_folder_paths):
    data = datas
    labels = labelss
    ID = IDs
    new_folder_path = working_directory_path / Path('{}'.format(new_folder_paths))
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
    assert (mask.GetPixelIDValue() == labels.GetPixelIDValue())

















