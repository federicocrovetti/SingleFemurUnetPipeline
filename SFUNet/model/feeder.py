#feeder
import numpy as np
import tensorflow as tf
from SFUNet.utils.slicer_utils import SliceDatasetLoader

class ImageFeeder(tf.keras.utils.Sequence):
    
    def __init__(self, batch_size, features_paths, labels_paths):
        self.batch_size = batch_size
        self.features_paths = features_paths
        self.labels_paths = labels_paths
        

    def __len__(self):
        return len(self.features_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_data = self.features_paths[i : i +self.batch_size]
        batch_input_labels = self.labels_paths[i : i +self.batch_size]
        
        x = np.zeros((self.batch_size,) + (256,256) + (1,), dtype="float32")
        for j in range(len(batch_input_data)):
            img, np_img = SliceDatasetLoader(batch_input_data[j], batch_input_labels[j])
            del(img)
            img = np_img['features']
            shift = (img + abs(np.min(img)))
            img = np.divide(shift, np.max(shift))
            img = np.expand_dims(img, axis=-1)
            x[j] = img
            
        y = np.zeros((self.batch_size,) + (256,256) + (1,), dtype="float32")
        for j in range(len(batch_input_labels)):
            lab, np_lab = SliceDatasetLoader(batch_input_data[j], batch_input_labels[j])
            del(lab)
            lab = np_lab['labels']
            lab = np.expand_dims(lab, axis=-1)
            y[j] = lab
            
            return x, y

