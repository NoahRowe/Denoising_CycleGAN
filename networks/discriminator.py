import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import (Input, Activation, Lambda,
                                     Conv1D, Conv1DTranspose,
                                     UpSampling1D, AveragePooling1D,
                                     Flatten, Dense)

##############################################################################
# Build the network
##############################################################################
def convolutional_classifier(input_tuple):
    '''
    TO DO:
    - Add input scaling
    '''
    
    activation = 'relu'
    
    input_signal = Input(shape=input_tuple, name="autoencoder_input")

    encode_i = Lambda(lambda X: tf.expand_dims(X, axis=-1), 
                      name="expand_dims_for_conv1d")(input_signal)

    # CONVOLUTIONAL LAYERS
    encode_h = Conv1D(filters=32, kernel_size=1, strides=1, padding='valid')(encode_i)
    encode_h = Activation(activation)(encode_h)

    encode_h = Conv1D(filters=32, kernel_size=9, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)

    encode_h = Conv1D(filters=32, kernel_size=17, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)

    encode_h = Conv1D(filters=32, kernel_size=33, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)

    encode_h = Conv1D(filters=32, kernel_size=17, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)
    
    encode_h = Conv1D(filters=32, kernel_size=9, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)
    
    encode_h = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)

    
    # Pass convolution output to a dense network
    dense_layer = Flatten()(encode_h)
    dense_layer = Dense(10, activation="relu")(dense_layer)
    output = Dense(1, activation="sigmoid", name="single_output")(dense_layer)
    
    #####################################################################################
    
    network = Model(input_signal, output)
    
    return network
