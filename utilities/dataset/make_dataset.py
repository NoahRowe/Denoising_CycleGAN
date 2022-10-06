import numpy as np

import tensorflow as tf
from tensorflow.data import Dataset

def make_dataset(list_X, list_y, batch_size=128, prefetch=150):
    '''
    Create a tensorflow dataset.
    '''

    ds = Dataset.from_tensor_slices((list_X, list_y))

    ds = ds.shuffle(len(list_X), reshuffle_each_iteration=True)

    ds = ds.flat_map(data_wrapper_numpy)
    
    ds = ds.shuffle(20000, reshuffle_each_iteration=True)
        
    ds = ds.batch(batch_size)#.prefetch(prefetch)

    return ds


def data_wrapper_numpy(filename_X, filename_Y):
    
    X, Y = tf.numpy_function(get_data_numpy, 
                             [filename_X, filename_Y],
                             (tf.float32, tf.float32))
    
    X.set_shape(tf.TensorShape([None, None]))
    Y.set_shape(tf.TensorShape([None, None]))
    
    return tf.data.Dataset.from_tensor_slices((X, Y))
    
def get_data_numpy(filename_X, filename_Y):

    X = np.load(filename_X)
    Y = np.load(filename_Y)

    return X.astype(np.float32), Y.astype(np.float32)