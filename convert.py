"""Conversion script"""
import argparse
import json
import os
import sys

from imports.convert import ConvertWrapper


def convert(settings='settings.json'):
    """Converts data from one format to another"""
    with open(settings, 'r') as file:
        config = json.load(file)
    cw = ConvertWrapper.from_json(config)
    cw.convert()
    print('Conversion finished')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('settings', help='File with settings for data conversion', nargs='?',
                      default='settings.json')
    parsed_args = args.parse_args(sys.argv[1:])

    settings_arg = parsed_args.settings

    if os.path.isfile(settings_arg):
        print('File -', settings_arg)
        convert(settings_arg)
