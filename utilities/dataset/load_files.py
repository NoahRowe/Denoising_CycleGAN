from pathlib import Path

def get_file_lists(base_dir):
    '''
    Function used to return a list of files to load for training, testing, 
    and validation.
    '''
    
    n_files = 1000  #max 1000
    X_filenames = [f"X_noisy_library_pulse_r{i}.npy" for i in range(n_files)] 
    Y_filenames = [f"y_clean_library_pulse_r{i}.npy" for i in range(n_files)]
    
    X_train = [str(Path(base_dir, "train", f)) for f in X_filenames]
    X_test = [str(Path(base_dir, "test", f)) for f in X_filenames]
    X_val = [str(Path(base_dir, "val", f)) for f in X_filenames]
    
    Y_train = [str(Path(base_dir, "train", f)) for f in Y_filenames]
    Y_test = [str(Path(base_dir, "test", f)) for f in Y_filenames]
    Y_val = [str(Path(base_dir, "val", f)) for f in Y_filenames]
    
    filenames = {
        "train": (X_train, Y_train),
        "test": (X_test, Y_test),
        "val": (X_val, Y_val)
    }
    
    return filenames