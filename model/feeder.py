# -*- coding: utf-8 -*-
#feeder
import numpy as np
import tensorflow as tf

class ImageFeeder(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, features_dataset, labels_dataset):
        self.batch_size = batch_size
        self.img_size = img_size
        self.features_dataset = features_dataset
        self.labels_dataset = labels_dataset

    def __len__(self):
        return len(self.labels_dataset) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_data = self.features_dataset[i : i + self.batch_size]
        batch_labels_data = self.labels_dataset[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j in range(len(batch_input_data)):
            img = batch_input_data[j]
            img = np.expand_dims(img, axis=2)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j in range(len(batch_labels_data)):
            label = batch_labels_data[j]
            label = np.expand_dims(label, axis=2)
            y[j] = label
        return x, y
    