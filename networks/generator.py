import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import (Input, Activation, Lambda,
                                     Conv1D, Conv1DTranspose,
                                     UpSampling1D, AveragePooling1D)

##############################################################################
# Build the network
##############################################################################

def autoencoder(input_tuple):
    '''
    TO DO:
    - Add scaling to front and back
    '''
    
    activation = 'relu'
    
    input_signal = Input(shape=input_tuple, name="autoencoder_input")

    encode_i = Lambda(lambda X: tf.expand_dims(X, axis=-1), 
                      name="expand_dims_for_conv1d")(input_signal)

    # ENCODER
    encode_h = Conv1D(filters=8, kernel_size=1, strides=1, padding='valid')(encode_i)
    encode_h = Activation(activation)(encode_h)

    encode_h = Conv1D(filters=16, kernel_size=9, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)

    encode_h = Conv1D(filters=32, kernel_size=17, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)

    encode_h = Conv1D(filters=64, kernel_size=33, strides=1, padding='valid')(encode_h)
    encode_h = Activation(activation)(encode_h)
    encode_h = AveragePooling1D(2)(encode_h)

    
    # MIDDLE (LATENT) LAYER
    encode_h = Conv1D(filters=32, kernel_size=33, strides=1, padding='valid')(encode_h)
    encode_o = Activation(activation, activity_regularizer=l1(0.01),
                          name="encoder_output")(encode_h)

    # DECODER
    decode_i = Conv1DTranspose(filters=32, kernel_size=33, strides=1,
                               padding='valid')(encode_o)
    decode_h = Activation(activation)(decode_i)

    decode_h = UpSampling1D(2)(decode_h)
    decode_h = Conv1DTranspose(filters=64, kernel_size=33, strides=1,
                               padding='valid')(decode_h)
    decode_h = Activation(activation)(decode_h)

    decode_h = UpSampling1D(2)(decode_h)
    decode_h = Conv1DTranspose(filters=32, kernel_size=17, strides=1,
                               padding='valid')(decode_h)
    decode_h = Activation(activation)(decode_h)

    decode_h = UpSampling1D(2)(decode_h)
    decode_h = Conv1DTranspose(filters=16, kernel_size=9, strides=1,
                               padding='valid')(decode_h)
    decode_h = Activation(activation)(decode_h)

    decode_o = Conv1D(filters=1, kernel_size=1, strides=1, padding='valid',
                      activity_regularizer=None)(decode_h)
    
    decode_o = Lambda(lambda X: tf.squeeze(X, axis=-1),
                  name="autoencoder_output")(decode_o)

    #####################################################################################

    network = Model(input_signal, decode_o)

    return network