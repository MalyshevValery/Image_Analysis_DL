"""Main launching script"""
import argparse
import json
import os
import sys
import traceback as tb
from multiprocessing import Process

import matplotlib.pyplot as plt

from imports.data.storages import DirectoryStorage
from imports.training import TrainingWrapper


def train(settings='settings.json', show_sample=False, predict=False, extended=False):
    """Train model

    :param settings: path to settings file
    :param show_sample: shows sample from input data if True
    :param predict: predicts and saves Test set if True
    :param extended: True to save extended version of settings
    """

    with open(settings, 'r') as file:
        config = json.load(file)
    tw = TrainingWrapper.from_json(config, 'Jobs/', settings)
    if show_sample:
        plt.axis('off')
        plt.imshow(tw.get_train_sample()[..., ::-1])
        plt.show(bbox_inches='tight')

    with open(os.path.join(tw.get_job_dir(), 'settings.json'), 'w') as file:
        json.dump(config, file, indent=2)
    if extended:
        with open(os.path.join(tw.get_job_dir(), 'settings_extended.json'), 'w') as file:
            json.dump(tw.to_json(), file, indent=2)

    tw.train(save_whole_model=True)
    tw.evaluate()

    if predict:
        pred_dir = os.path.join(tw.get_job_dir(), 'predicted')
        tw.predict_save_test(DirectoryStorage(pred_dir, mode='w'))


def check(settings='settings.json', **kwargs):
    """Checks if settings are valid"""
    with open(settings, 'r') as file:
        config = json.load(file)
    tw = TrainingWrapper.from_json(config, 'Jobs/test_', settings)
    tw.check()
    os.rmdir(tw.get_job_dir())
    print('Everything is OK')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('command', help='Command to execute', choices=['train', 'check'])
    args.add_argument('settings', help='File with settings or directory with different settings', nargs='?',
                      default='settings.json')
    args.add_argument('-p', help='Save predicted test set to job dir', action='store_true', dest='predict')
    args.add_argument('-s', help='Show sample', action='store_true', dest='show_sample')
    args.add_argument('-e', help='Save extended settings', action='store_true', dest='extended')
    parsed_args = args.parse_args(sys.argv[1:])

    kwargs = vars(parsed_args)
    settings_arg = parsed_args.settings
    command = parsed_args.command
    if command == 'train':
        command = train
    elif command == 'check':
        command = check
    else:
        raise ValueError('Unknown command ' + command)
    del parsed_args.command

    if os.path.isfile(settings_arg):
        print('File -', settings_arg)
        command(**vars(parsed_args))
    elif os.path.isdir(settings_arg):
        print('Dir -', settings_arg)
        for f in os.listdir(settings_arg):
            settings_file = os.path.join(settings_arg, f)
            print('File -', settings_file)
            try:
                kwargs['settings'] = settings_file
                proc = Process(target=command, kwargs=kwargs)
                proc.start()
                proc.join()
            except Exception as e:
                tb.print_exc()
