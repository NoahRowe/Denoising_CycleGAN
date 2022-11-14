import warnings

import numpy as np


def add_vscale(pulse_arr, low, high=None):
    '''
    Scale an array of pulses by amplitude.
    Each scale is drawn from a random uniform distribution.
    '''
    # If desired, can have a constant scale by setting high=None.
    if high is None:
        vertical_scale = low
    else:
        vertical_scale = np.random.uniform(low=low, high=high, size=(pulse_arr.shape[0], 1))

    pulse_arr *= vertical_scale

    
    
def add_hshift(pulse_arr, low, high, abs_shift):
    '''
    Shift an array of pulses horizontally.
    Each shift is drawn from a random uniform distribution.

    The abs_shift argument reshifts pulses to zero before applying the shift,
    ensuring that shifts are absolute (not relative) to the existing shift.
    '''

    # First check that maximum value (first instance) is the same as
    # last value in array (i.e., flat top).
    ind_rows = np.arange(pulse_arr.shape[0])
    ind_cols = np.argmax(pulse_arr, axis=1)
    if not np.allclose(pulse_arr[ind_rows, ind_cols], pulse_arr[:, -1]):
        raise ValueError("Attempting to horizontally shift array "
                         "that may have noise or exponential decay. "
                         "This is not supported.")

    # If desired, can have a constant shift.
    if high is None:
        high = low + 1
    
    # Get the current shift, i.e. rise start position (typically not 0),
    # which is needed to perform an absolute shift.
    if abs_shift:
        if low > 0 or high > 0:
            warnings.warn("Shifting an array of pulses to the left "
                          "when specifying that the shift is absolute. "
                          "This will wrap the pulses and may not be "
                          "your intention.")
        rise_start = (pulse_arr.shape[1] - np.argmin(pulse_arr[:, ::-1], axis=1) - 1)
    else:
        rise_start = 0

    # Generate an array of random horizontal shifts.
    horizontal_shift = np.random.randint(low=low, high=high,
                                         size=pulse_arr.shape[0])

    # Compute the total shift, first to the left (positive) to undo the shift (i.e. move rise start to zero),
    # then to the right (negative) to get to the new shift point.
    total_shift = rise_start + horizontal_shift

    # Apply random horizontal shift to each pulse.
    for i, hshift_i in enumerate(total_shift):
        if hshift_i > 0:
            # Shifts the array to the left.
            pulse_arr[i][..., 0:-hshift_i] = pulse_arr[i][..., hshift_i:]
            # Ensures that the beginning/end are all the same value.
            pulse_arr[i][..., -hshift_i:] = pulse_arr[i][..., -1:]

        elif hshift_i < 0:
            # Shifts the array to the right.
            pulse_arr[i][..., -hshift_i:] = pulse_arr[i][..., 0:hshift_i]
            # Ensures that the beginning/end are all the same value.
            pulse_arr[i][..., 0:-hshift_i] = pulse_arr[i][..., 0:1]

            

def add_vshift(pulse_arr, low, high=None):
    '''
    Shift an array of pulses vertically.
    Each shift is drawn from a random uniform distribution.
    '''
    # If desired, can have a constant shift.
    if high is None:
        vertical_shift = low
    else:
        vertical_shift = np.random.uniform(low=low, high=high, size=(pulse_arr.shape[0], 1))

    pulse_arr += vertical_shift

