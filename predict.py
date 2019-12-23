"""Main predicting script"""
import argparse
import json
import os
import sys
import traceback as tb
from multiprocessing import Process

from imports.data.storages import DirectoryStorage
from imports.predict import PredictWrapper


def predict(job_dir, image_dir):
    """Predict masks from images in provided directory

    :param job_dir: directory with job (settings and weights are necessary
    :param image_dir: directory with images to predict
    :return:
    """
    filename = os.path.join(job_dir, 'settings_extended.json')
    if not os.path.isfile(filename):
        filename = os.path.join(job_dir, 'settings.json')
    with open(filename, 'r') as file:
        config = json.load(file)

    pw = PredictWrapper.from_json(config, job_dir)
    color_mode = config['loader']['images'].get('color_transform', 'none')
    basename = os.path.basename(os.path.abspath(image_dir))
    pred_dir = os.path.join(job_dir, 'predicted_' + basename)
    pw.predict_storage(DirectoryStorage(image_dir, color_transform=color_mode), DirectoryStorage(pred_dir, mode='w'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('job', help='Directory with trained job/jobs to load and predict')
    args.add_argument('images', help='Directory with images to predict')
    args.add_argument('-m', help='Treat job dir as directory with multiple job directories', action='store_true',
                      dest='multiple')
    parsed_args = args.parse_args(sys.argv[1:])

    if not os.path.isdir(parsed_args.job):
        raise Exception(parsed_args.job + ' is not a directory')

    if parsed_args.multiple:
        for dir_ in os.listdir(parsed_args.job):
            path = os.path.join(parsed_args.job, dir_)
            if not os.path.isdir(path):
                print(dir_, "is not a dir")
                continue
            print('Dir -', path)
            try:
                proc = Process(target=predict, args=(path, parsed_args.images))
                proc.start()
                proc.join()
            except Exception as e:
                tb.print_exc()
    else:
        try:
            predict(parsed_args.job, parsed_args.images)
        except Exception as e:
            tb.print_exc()
