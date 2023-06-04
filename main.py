import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

def model(shape, classes):
    input = keras.Input(shape=shape)

    a = layers.Rescaling(1.0 / 255)(input)
    a = layers.Conv2D(128, 3, strides=2, padding='same')(a)
    a = layers.BatchNormalization()(a)
    a = layers.Activation("relu")(a)

    block = a

    for x in [256, 512, 728]:
        a = layers.Activation("relu")(a)
        a = layers.SeparableConv2D(x, 3, padding='same')(a)
        a = layers.BatchNormalization()(a)
        a = layers.Activation("relu")(a)
        a = layers.SeparableConv2D(x, 3, padding='same')(a)
        a = layers.BatchNormalization()(a)
        a = layers.MaxPooling2D(3, strides=2, padding='same')(a)

        r = layers.Conv2D(x, 1, strides=2, padding='same')(block)

        a = layers.add([a, r])
        block = a

    a = layers.SeparableConv2D(1024, 3, padding='same')(a)
    a = layers.BatchNormalization()(a)
    a = layers.Activation("relu")(a)
    a = layers.GlobalAveragePooling2D()(a)

    #several emotions, so use softmax
    if classes >= 2:
        activation = 'softmax'
        units = classes

    a = layers.Dropout(0.5)(a)
    output = layers.Dense(units, activation=activation)(x)
    return keras.Model(input, output)