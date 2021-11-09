"""2DUnet"""


import numpy as np
from pathlib import Path
from _dataload_and_preprocessing_ import DataLoad 
import tensorflow as tf
import random
from IPython.display import Image, display
import PIL
from PIL import ImageOps


patients = []
data = []
masks = []
data_folders = []
basepath = Path(r'D:/Fedz/Pattern_Recognition_Project/HipOp_OK_resampled') 


files_in_basepath = basepath.iterdir()
for item in files_in_basepath:
    if item.is_dir():
        if not item.name == '__pycache__':
            print(item.name)
            data_folders.append(item.name)
            path = basepath / '{}'.format(item.name)
            patients.append(path)
            
files_in_basepath = basepath.iterdir()
for item in files_in_basepath:
    if item.is_dir():
        if not item.name == '__pycache__':
            for elem in item.iterdir():
                if "Data" in elem.name:
                    data.append(elem)
                elif "Segmentation" in elem.name:
                    masks.append(elem)

dataset, dataset_array, metadata_images, metadata_masks = DataLoad(data, masks)

trial_dst = {'features': [], 'labels':[]}
#unified dict with all images and masks sliced into (x,x) numpy arrays

for i in range(len(dataset_array['features'])):
    for j in range(len(dataset_array['features'][i])):
        trial_dst['features'].append(dataset_array['features'][i][:,:,j])
        
for i in range(len(dataset_array['labels'])):
    for j in range(len(dataset_array['labels'][i])):
        trial_dst['labels'].append(dataset_array['labels'][i][:,:,j])
        
        

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
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j in range(len(batch_input_data)):
            img = np.expand_dims(batch_input_data[j], axis=2)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j in range(len(batch_labels_data)):
            img = batch_labels_data[j]
            y[j] = np.expand_dims(img, 2)
        return x, y


img_size = (280, 280)
num_classes = 2

def get_model(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (1,))

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  

    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual]) 
        previous_block_activation = x


    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling2D(2)(x)

        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])
        previous_block_activation = x

    x = tf.keras.layers.Conv2D(32, 9, padding="valid")(x)    
    outputs = tf.keras.layers.Conv2D(num_classes, 2, activation="softmax", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


tf.keras.backend.clear_session()
model = get_model(img_size, num_classes)
model.summary()



tot_train_samples = 2000
random.Random(656).shuffle(trial_dst['features'])
random.Random(656).shuffle(trial_dst['labels'])

train_dst_features = trial_dst['features'][:tot_train_samples]
train_dst_labels = trial_dst['labels'][:tot_train_samples]

val_dst_features = trial_dst['features'][tot_train_samples:]
val_dst_labels = trial_dst['labels'][tot_train_samples:]

train_dst = ImageFeeder(15, (280,280), train_dst_features, train_dst_labels)
val_dst = ImageFeeder(15, (280,280), val_dst_features, val_dst_labels)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",  metrics=['accuracy'])

checkpoint_filepath = 'D:/Fedz/Pattern_Recognition_Project/HipOp_OK/cbks.h5'

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    save_weights_only=True, monitor='val_accuracy',
    mode='max', save_best_only=True)
]


epochs = 30

model.fit(train_dst, epochs=epochs, validation_data = val_dst, callbacks=callbacks) 

















