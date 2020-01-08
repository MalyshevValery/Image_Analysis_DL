"""Training script"""
import argparse
import json
import os
import sys
import traceback as tb
from multiprocessing import Process
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from imports.data.storages import DirectoryStorage
from imports.train import TrainWrapper
from imports.optim import optimize, get_space_for_json


def train_single(settings='settings.json', predict=False, extended=False, prefix='Jobs/'):
    """Train model

        :param settings: path to settings file
        :param predict: predicts and saves Test set if True
        :param extended: True to save extended version of settings
        :param prefix: directory for jobs, adds custom prefixes for params
        """

    if settings is str:
        with open(settings, 'r') as file:
            config = json.load(file)
            tw = TrainWrapper.from_json(config, prefix, settings)
    else:
        config = settings
        tw = TrainWrapper.from_json(config, prefix)

    with open(os.path.join(tw.get_job_dir(), 'settings.json'), 'w') as file:
        json.dump(config, file, indent=2)
    if extended:
        with open(os.path.join(tw.get_job_dir(), 'settings_extended.json'), 'w') as file:
            json.dump(tw.to_json(), file, indent=2)

    tw.train()
    ret_val = tw.evaluate()

    if predict:
        pred_dir = os.path.join(tw.get_job_dir(), 'predicted')
        tw.predict_save_test(DirectoryStorage(pred_dir, mode='w'))
    print('TEST EVALUATION METRIC: ', ret_val)
    return ret_val


def train(settings='settings.json', predict=False, extended=False):
    with open(settings, 'r') as file:
        config = json.load(file)
    space = get_space_for_json(config)
    if len(space) == 0:
        return train_single(config, predict, extended)
    best = optimize(lambda new_config, param_string: train_single(new_config, predict, extended, 'Jobs/' + param_string)
                    , config, space)
    print('BEST RESULTS: ', best)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('settings', help='File with settings or directory with different settings', nargs='?',
                      default='settings.json')
    args.add_argument('-p', help='Save predicted test set to job dir', action='store_true', dest='predict')
    args.add_argument('-e', help='Save extended settings', action='store_true', dest='extended')
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
