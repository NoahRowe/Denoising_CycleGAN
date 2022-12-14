import warnings

import tensorflow as tf

hline = f"{'':-<80}"

def configure_tensorflow(eager=False,
                         gpu_index=None,
                         allow_growth_gpu=False,
                         memory_limit_MB=4000):
    '''
    Code to sort out gpu usage when training models.
    '''
    # Get available GPUs.
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(hline)
        print("GPU information:")
        for gpu in gpus:
            print(f"  GPU available: {gpu}")
        print("")

        # Set the GPU to use, if applicable.
        if gpu_index is not None:
            if gpu_index >= 0:
                print(f"Using GPU {gpu_index}.")
                tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            else:
                print("Disabling GPUs.")
                tf.config.set_visible_devices([], 'GPU')

        print(hline)

        # If not allowed to dynamically grow, then use hardcoded numbers that work.
        # I know, not great... Will fix this one day.
        if allow_growth_gpu:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            for gpu in gpus:
                log_dev_config = [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_MB)]
                tf.config.set_logical_device_configuration(gpu, log_dev_config)
    else:
        warnings.warn("No GPUs detected.", RuntimeWarning)

    # Turn off eager execution.
    if not eager:
        tf.compat.v1.disable_eager_execution()

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    # Print some information on TensorFlow.
    print(hline)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Eager execution enabled: {tf.compat.v1.executing_eagerly()}")
    print(f"{len(gpus)} physical GPU(s), {len(logical_gpus)} logical GPU(s)")
    print(hline)
