"""Check script"""
import argparse
import json
import os
import sys
import traceback as tb
from multiprocessing import Process

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from imports.train import TrainWrapper


def check(settings='settings.json', show_sample=False):
    """Checks if settings are valid

    :param settings: path to JSON file with setup
    :param show_sample: shows sample from input data if True
    """
    with open(settings, 'r') as file:
        config = json.load(file)
    tw = TrainWrapper.from_json(config, 'Jobs/test_', settings)
    try:
        if show_sample:
            plt.axis('off')
            plt.imshow(tw.get_train_sample()[..., ::-1])
            plt.show(bbox_inches='tight')
        tw.check()
    except Exception as exc:
        raise exc
    finally:
        os.rmdir(tw.get_job_dir())
    print('Everything is OK')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('settings', help='File with settings or directory with different settings', nargs='?',
                      default='settings.json')
    args.add_argument('-s', help='Show sample', action='store_true', dest='show_sample')
    parsed_args = args.parse_args(sys.argv[1:])

    kwargs = vars(parsed_args)
    settings_arg = parsed_args.settings

    if os.path.isfile(settings_arg):
        print('File -', settings_arg)
        check(**vars(parsed_args))
    elif os.path.isdir(settings_arg):
        print('Dir -', settings_arg)
        for f in os.listdir(settings_arg):
            settings_file = os.path.join(settings_arg, f)
            print('File -', settings_file)
            try:
                kwargs['settings'] = settings_file
                proc = Process(target=check, kwargs=kwargs)
                proc.start()
                proc.join()
            except Exception as e:
                tb.print_exc()
