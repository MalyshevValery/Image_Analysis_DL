"""Main training script"""
import argparse
import json
import os
import sys
import traceback as tb
from multiprocessing import Process

import matplotlib.pyplot as plt

from imports.data.storages import DirectoryStorage
from imports.trainingwrapper import TrainingWrapper


def train(settings='settings.json', show_sample=False, predict=False):
    """Train model

    :param settings: path to settings file
    :param show_sample: shows sample from input data if True
    :param predict: predicts and saves Test set if True
    """

    with open(settings, 'r') as file:
        config = json.load(file)
    tw = TrainingWrapper.from_json(config, 'Jobs/', settings)
    if show_sample:
        plt.axis('off')
        plt.imshow(tw.get_train_sample())
        plt.show(bbox_inches='tight')

    tw.train(save_whole_model=True)
    tw.evaluate()

    if predict:
        pred_dir = os.path.join(tw._job_dir, 'predicted')
        tw.predict_save_test(DirectoryStorage(pred_dir, mode='w'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('settings', help='File with settings or directory with different settings', nargs='?',
                      default='settings.json')
    args.add_argument('-p', help='Save predicted test set to job dir', action='store_true', dest='predict')
    args.add_argument('-s', help='Show sample', action='store_true', dest='show_sample')
    parsed_args = args.parse_args(sys.argv[1:])

    kwargs = vars(parsed_args)
    settings_arg = parsed_args.settings
    if os.path.isfile(settings_arg):
        print('File -', settings_arg)
        train(**vars(parsed_args))
    elif os.path.isdir(settings_arg):
        print('Dir -', settings_arg)
        for f in os.listdir(settings_arg):
            settings_file = os.path.join(settings_arg, f)
            print('File -', settings_file)
            try:
                kwargs['settings'] = settings_file
                proc = Process(target=train, kwargs=kwargs)
                proc.start()
                proc.join()
            except Exception as e:
                tb.print_exc()
