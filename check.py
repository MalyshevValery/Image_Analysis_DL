"""Check script"""
import argparse
import json
import os
import sys
import traceback as tb
from multiprocessing import Process

from imports.train import TrainWrapper


def check(settings='settings.json'):
    """Checks if settings are valid"""
    with open(settings, 'r') as file:
        config = json.load(file)
    tw = TrainWrapper.from_json(config, 'Jobs/test_', settings)
    tw.check()
    os.rmdir(tw.get_job_dir())
    print('Everything is OK')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('settings', help='File with settings or directory with different settings', nargs='?',
                      default='settings.json')
    parsed_args = args.parse_args(sys.argv[1:])

    kwargs = vars(parsed_args)
    settings_arg = parsed_args.settings

    if os.path.isfile(settings_arg):
        print('File -', settings_arg)
        check()
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
