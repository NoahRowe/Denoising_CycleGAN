from pathlib import Path
import warnings

import numpy as np

################################################################################

def real_noise(pulse_arr, noise_arr_list, mask=None, **kwargs):
    '''
    Add real noise to an array sampled from a list of numpy array(s).
    Should be memory-mapped if noise arrays are large.
    '''
    # Check if list of files or list of numpy arrays already.
    if all(isinstance(f, (str, Path)) for f in noise_arr_list):
        noise_arr_list = [np.load(f, mmap_mode='r') for f in noise_arr_list]
    elif all(isinstance(arr, np.ndarray) for arr in noise_arr_list):
        # Flush the memory-mapped arrays.
        for arr in noise_arr_list:
            arr.flush()
    else:
        raise ValueError("noise_arr_list must be either a "
                         "list of numpy arrays or a "
                         "list of file paths to numpy arrays.")

    # Get the number of entries in the noise files and in the array.
    noise_entries = sum(arr.shape[0] for arr in noise_arr_list)
    pulse_entries = pulse_arr.shape[0]
    
    # Choose random indexes for the array of pulses.
    # Set the entries drawn to False so that they are not drawn again.
    indexes_all = np.arange(0, noise_entries)
    if mask is not None:
        indexes = np.random.choice(indexes_all[mask], replace=False, size=pulse_entries)
        mask[indexes] = False

    # Load in the selected pulses by entry.
    noise_arr = load_pulses_by_index(noise_arr_list,
                                     indexes=indexes,
                                     nsamples=pulse_arr.shape[-1])

    # Rescale and add the noise.
    rescale_noise(noise_arr, **kwargs)
    pulse_arr += noise_arr

################################################################################

def load_pulses_by_index(np_arrays, indexes, nsamples=None):
    '''
    Load and process certain entries of a list of numpy arrays into one array.
    '''
    # Counters to keep track of last index of the previous arrays.
    s = 0
    n = 0

    indexes_list = []
    for arr in np_arrays:
        # Select only the indexes for the given file.
        mask = np.logical_and((indexes >= s), (indexes < s + arr.shape[0]))
        indexes_arr = indexes[mask] - s
        indexes_arr.sort()
        indexes_list.append(indexes_arr)

        # Increment the counters.
        s += arr.shape[0]
        n += indexes_arr.shape[0]

    arr_new_tuple = tuple(arr[iarr, :nsamples] for arr, iarr in zip(np_arrays, indexes_list))
    arr_new = np.concatenate(arr_new_tuple, axis=0)

    return arr_new


def rescale_noise(noise_arr, sigma_loc, sigma_scale=None):
    '''
    Given an array, rescale it to a random standard deviation.
    
    '''
    # Normalize to mean zero and unit variance.
    noise_arr -= noise_arr.mean(axis=1, keepdims=True)
    noise_arr /= noise_arr.std(axis=1, keepdims=True)

    # Generate the standard deviation from a random uniform distribution.
    sigma = np.random.normal(loc=sigma_loc, scale=sigma_scale,
                             size=(noise_arr.shape[0], 1))

    noise_arr *= sigma


