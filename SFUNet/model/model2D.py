# -*- coding: utf-8 -*-
#model

import tensorflow as tf


def model2D(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (1,))

    x = tf.keras.layers.Conv2D(16, 5, strides=(1,1), padding="same", 
                               data_format='channels_last', dilation_rate=(2,2), activation = "relu", dtype = tf.float32)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    input_block = x  
    x = tf.keras.layers.MaxPooling2D(2, strides=(2, 2), padding="valid", dtype = tf.float32)(x)
    intermediate_blocks = []
    
    for filters in [32, 64, 128, 256]:
        x = tf.keras.layers.Conv2D(filters, 3, strides=(1, 1), padding="same", 
                                            data_format='channels_last', dilation_rate=(2,2), activation = "relu", dtype = tf.float32)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, strides=(1, 1), padding="same", 
                                            data_format='channels_last', dilation_rate=(2,2), activation = "relu", dtype = tf.float32)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        intermediate_blocks.append(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=(2, 2), padding="valid", data_format='channels_last', dtype = tf.float32)(x)
        

    x = tf.keras.layers.Conv2D(512, 3, strides=(1, 1), padding="same", 
                                            data_format='channels_last', dilation_rate=(2,2), activation = "relu", dtype = tf.float32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    indexes = {'256': 3, '128': 2, '64': 1, '32': 0}
    
    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Conv2DTranspose(filters, 3, strides = (2,2), padding="same", 
                                            data_format='channels_last', activation = "relu", dtype = tf.float32)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Concatenate(axis = -1, dtype = tf.float32)([x, intermediate_blocks[indexes['{}'.format(filters)]]])
        
        
    x = tf.keras.layers.Conv2DTranspose(32, 5, strides = (2,2), padding="same", 
                                            data_format='channels_last', activation = "relu", dtype = tf.float32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate(axis = -1, dtype = tf.float32)([x, input_block])
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="sigmoid", padding="same", data_format='channels_last', dtype = tf.float32)(x)

    model = tf.keras.Model(inputs, outputs)
    return model


