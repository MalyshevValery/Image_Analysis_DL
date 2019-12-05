import os
import tensorflow as tf
import json


def gpu_setup(gpu_settings):
    if tf.test.is_gpu_available():
        available_gpu = tf.config.experimental.list_physical_devices('GPU')
        for gpu in available_gpu:
            tf.config.experimental.set_memory_growth(gpu, True)
        if not type(gpu_settings['gpu']) is list:
            gpu_settings['gpu'] = [gpu_settings['gpu']]
        for i in gpu_settings['gpu']:
            assert len(available_gpu) > i
            assert i >= 0
        gpus = [available_gpu[i] for i in gpu_settings['gpu']]
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
