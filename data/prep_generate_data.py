#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from preprocess.data_augmentation import augment_data_train_test_split
from preprocess.add_noise import real_noise

# Set seed for reproducibility 
# Will produce the same dataset (including shuffle order) for every run.
np.random.seed(1234567)

##########################################################################
# Specify noise generation method and parameters.

# Parameters of the noise addition.
vscale_params = {'low': 0.9, 'high': 1.1}
hshift_params = {'low': -1050, 'high': -950, 'abs_shift': True}
vshift_params = {'low': -0.1, 'high': 0.1}


# Use noise files.
noise_file_dir = '/scratch/manderson/numpy/pulses/noise'
noise_file_list = [str(f) for f in Path(noise_file_dir).glob('RUN1000019/data*output.npy')]

np_arrays = [np.load(f, mmap_mode='r') for f in noise_file_list]
mask = np.ones(sum(n.shape[0] for n in np_arrays), dtype=bool)

noise_function = real_noise
noise_params = {'noise_arr_list': np_arrays,
                'mask': mask,
                'sigma_loc': 0.08,
                'sigma_scale':0.005,}

##########################################################################
# Collect all of the arguments.
# Pass to the function to augment the data.
# Processes the original noise arrays.

n_dup = 1000
augment_data_params = {'n_dup': n_dup,
                       'noise_function': noise_function,
                       'noise_params': noise_params,
                       'vertical_scale_params': vscale_params,
                       'horizontal_shift_params': hshift_params,
                       'vertical_shift_params': vshift_params}

# Directory from where to load the numpy files.
numpy_path = '/home/tye/data/numpy/library/library_pulse.npy'
numpy_file_dir = '/home/tye/data/numpy/fake/data1002'


# Augment the training data.
augment_data_train_test_split(numpy_filepath_list=(numpy_path,),
                              numpy_file_dir=numpy_file_dir,
                              **augment_data_params)


