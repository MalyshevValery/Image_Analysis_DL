"""GPU setup"""
import tensorflow as tf


def gpu_setup(gpu=-1):
    """GPU setup

    :param gpu: list of GPUs that will be used
    :return:
    """
    if gpu == -1:
        tf.config.experimental.set_visible_devices([])
        return

    if tf.test.is_gpu_available():
        available_gpu = tf.config.experimental.list_physical_devices('GPU')
        for gpu in available_gpu:
            tf.config.experimental.set_memory_growth(gpu, True)
        if not type(gpu) is list:
            gpu = [gpu]
        gpu = [int(g) for g in gpu]
        assert max(gpu) < len(available_gpu)
        gpu_devices = [available_gpu[i] for i in gpu]
        tf.config.experimental.set_visible_devices(gpu_devices, 'GPU')
        tf.config.experimental.set_memory_growth(True)
