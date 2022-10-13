import SimpleITK as sitk
import numpy as np
from SFUNet.utils.data_pipeline import Halve, BedRemoval, Thresholding, Crop
import pytest
import hypothesis as hp
from hypothesis import given, settings
import hypothesis.strategies as st

#STRATEGIES

legitimate_chars = st.characters(whitelist_categories=('Lu','Ll')
                                    ,min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=5,
                            max_size=30)


@st.composite
def dataset_generator(draw):
  """
  Strategy for the generation of a dict object with a 3D image for features and labels keys.
  The size of them is the one with which the program will work with (512,512,z).
  """
  dataset = {'features' : [], 'labels' : []}
  side = draw(st.lists(st.integers(0,1), min_size = 1, max_size = 3))
  for i in range(len(side)):
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([1., 0., 0., 0., 1., 0., 0., 0., 1.])
        
    features = np.random.randint(200, size=(40, 512, 512))
    if side[i] == 0:
      label = np.zeros((40, 512, 512))
      label[:, : , 0:256] = np.full((40, 512 , 256), 1)
    else:
      label = np.zeros((40, 512, 512))
      label[:, : , 256:512] = np.full((40, 512 , 256), 1)
      image = sitk.GetImageFromArray(features)
      labels = sitk.GetImageFromArray(label)
      image.SetOrigin(origin)
      image.SetSpacing(spacing)
      image.SetDirection(direction)
      labels.SetOrigin(origin)
      labels.SetSpacing(spacing)
      labels.SetDirection(direction)
      dataset['features'].append(image)
      dataset['labels'].append(labels)
    return (dataset, side) 


@st.composite
def BedImageGenerator(draw):
  """
  Strategy for the generation of a dict object with a 3D image for features and labels keys.
  The size of them is the one with which the program will work with (512,512,z).
  Those images contain the main volume and a thin separated element.
  """
  dataset = {'features' : [], 'labels' : []}
  dataset_confront = {'features' : [], 'labels' : []}
  side = draw(st.lists(st.integers(0,1), min_size = 1, max_size = 3))
  for i in range(len(side)):
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([1., 0., 0., 0., 1., 0., 0., 0., 1.])
    feat_arr = np.zeros((40, 512, 512))
    feat_arr[:, 200:400, 200:350] = np.full((40, 200, 150), 1)
    feat_confront = sitk.GetImageFromArray(feat_arr)
    feat_confront.SetOrigin(origin)
    feat_confront.SetSpacing(spacing)
    feat_confront.SetDirection(direction)
    dataset_confront['features'].append(feat_confront)
    height = draw(st.integers(min_value = 2, max_value = 20))
    y = draw(st.integers(min_value = 0, max_value = (200-height)))
    feat_arr[:, y:(y+height), 100:450] = np.full((40, height, 350), 1)
       
    if side[i] == 0:
      label = np.zeros((40, 512, 512))
      label[:, : , 0:256] = np.full((40, 512 , 256), 1)
    else:
      label = np.zeros((40, 512, 512))
      label[:, : , 256:512] = np.full((40, 512 , 256), 1)
      image = sitk.GetImageFromArray(feat_arr)
      labels = sitk.GetImageFromArray(label)
      image.SetOrigin(origin)
      image.SetSpacing(spacing)
      image.SetDirection(direction)
      labels.SetOrigin(origin)
      labels.SetSpacing(spacing)
      labels.SetDirection(direction)
      dataset['features'].append(image)
      dataset['labels'].append(labels)
      dataset_confront['features'].append(labels)
    return (dataset, dataset_confront, side) 


@st.composite
def crop_set_generator(draw):
  """
  Strategy for the generation of a dict object with a 3D image for features and labels keys. 
  Images contained in this dict, both for features and labels, are of the shape (256,512,z),
  containing an object with random but limited size. 
  This object will be at maximum (256,256) along x & y axis (x dimension is fixed to 256), and will be placed at a random
  height on y axis.
  
  In addition, this strategy generates two lists of arrays:
    - shapes contain the arrays composed by minimum and maximum points of the image extension for each axis (z, y, x), for each slice;
    - dims contain the shape of the 3D image.
  """
  dataset = {'features' : [], 'labels' : []}
  dims = []
  shapes = []
  side = draw(st.lists(st.integers(0,1), min_size = 1, max_size = 3))
  for i in range(len(side)):
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([1., 0., 0., 0., 1., 0., 0., 0., 1.])
    y_min = draw(st.integers(0, 254))
    y_max = draw(st.integers(y_min + 257, 512))
    z_min = draw(st.integers(0, 38))
    z_max = draw(st.integers(z_min + 1, 40))
    shape = (z_min, z_max, y_min, y_max, 0, 256)
    #print(shape)
    features = np.zeros((40, 512, 256))
    features[shape[0] : shape[1], shape[2]:shape[3], shape[4]:shape[5]] = np.full((np.abs(shape[1] - shape[0]), np.abs(shape[3] - shape[2]), np.abs(shape[5] - shape[4])), 1)
    if side[i] == 0:
      label = np.ones((40, 512, 256))
    else:
      label = np.ones((40, 512, 256))
      image = sitk.GetImageFromArray(features)
      labels = sitk.GetImageFromArray(label)
      image.SetOrigin(origin)
      image.SetSpacing(spacing)
      image.SetDirection(direction)
      labels.SetOrigin(origin)
      labels.SetSpacing(spacing)
      labels.SetDirection(direction)
      dataset['features'].append(image)
      dataset['labels'].append(labels)
      slices_shapes = []
    for j in range(40):
      if any(image[:,:,j]):
        print(shape)
        slices_shapes.append(np.array((shape[4], shape[5], shape[2], shape[3], shape[0], shape[1])))
      else:
        slices_shapes.append([7000, 7000, 7000, 7000, 7000, 7000])
        shapes.append(slices_shapes)
        
      dims_i = (np.abs(shape[1] - shape[0]), np.abs(shape[3] - shape[2]), np.abs(shape[5] - shape[4]))
      dims.append(dims_i)
    return (dataset, shapes, dims)



@given(dataset_generator())
@settings(suppress_health_check=[hp.HealthCheck.too_slow], max_examples=50, deadline = None)
def test_Halve(gen):
  dst = gen[0]
  side = gen[1]
  halve_dst = Halve(dst, side, train = True)
  for i in range(len(halve_dst['features'])):
    assert(halve_dst['features'][i].GetSize()[:2] == (256, 512))
    assert(halve_dst['labels'][i].GetSize()[:2] == (256, 512))
    assert(any(halve_dst['labels'][i][:,:,:]))
        
    
@given(dataset_generator(), st.integers(0, 50), st.integers(50, 200))
@settings(suppress_health_check=[hp.HealthCheck.too_slow], max_examples=50, deadline = None)
def test_Thresholding(gen, low_threshold, high_threshold):
  dst = gen[0]
  threshold = [low_threshold, high_threshold]
  thres_dst = Thresholding(dst, threshold)
  for i in range(len(thres_dst['features'])):
    img_arr = sitk.GetArrayFromImage(thres_dst['features'][i])
    assert(np.any((img_arr == 0)) or np.any((img_arr == 1)))

@given(BedImageGenerator())
@settings(suppress_health_check=[hp.HealthCheck.too_slow], max_examples=50, deadline = None)
def test_BedRemoval(gen):
  dst = gen[0]
  no_bed = BedRemoval(dst)
  for i in range(len(gen[0]['features'])):
    arr_nobed_img = sitk.GetArrayFromImage(no_bed['features'][i])
    assert(np.sum(arr_nobed_img[0:512, 0:199, 0:40]) == 0)



@given(crop_set_generator(), text_strategy)
@settings(suppress_health_check=[hp.HealthCheck.too_slow], max_examples=2, deadline = None)
def test_Crop(dst_gen, IDstr):
  new_folder_path = 'g'
  dataset = dst_gen[0]
  bbox_grouped = dst_gen[1]
  ID = IDstr
   
  crop_dst = Crop(dataset, bbox_grouped, ID, new_folder_path)
    
  for i in range(len(crop_dst['features'])):
    assert(crop_dst['features'][i].GetSize()[0] == 256)
    assert(crop_dst['features'][i].GetSize()[1] == 256)
    assert(crop_dst['labels'][i].GetSize()[0] == 256)
    assert(crop_dst['labels'][i].GetSize()[1] == 256)
