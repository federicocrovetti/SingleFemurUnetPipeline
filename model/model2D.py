# -*- coding: utf-8 -*-
#model

import tensorflow as tf


def model2D(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (1,))
    print(inputs.shape)
    print(tf.rank(inputs))

    x = tf.keras.layers.Conv2D(16, 3, strides=(1,1), padding="same", 
                               data_format='channels_last', activation = "relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    input_block = x  
    x = tf.keras.layers.MaxPooling2D(2, strides=(2, 2), padding="valid")(x)
    intermediate_blocks = []
    
    for filters in [32, 64, 128]:
        x = tf.keras.layers.SeparableConv2D(filters, 3, strides=(1, 1), padding="same", 
                                            data_format='channels_last', activation = "relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, strides=(1, 1), padding="same", 
                                            data_format='channels_last', activation = "relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        intermediate_blocks.append(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=(2, 2), padding="valid")(x)
        

    x = tf.keras.layers.SeparableConv2D(256, 3, strides=(1, 1), padding="same", 
                                            data_format='channels_last', activation = "relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    indexes = {'128': 2, '64': 1, '32':0}
    for filters in [128, 64, 32]:
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides = 2, padding="same", 
                                            data_format='channels_last', activation = "relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.concat([x, intermediate_blocks[indexes['{}'.format(filters)]]], -1)
        
        
    x = tf.keras.layers.Conv2DTranspose(16, 2, strides = 2, padding="valid", 
                                            data_format='channels_last', activation = "relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.concat([x, input_block], -1)
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


