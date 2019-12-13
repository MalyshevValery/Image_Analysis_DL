"""Main predicting script"""
import argparse
import os
import sys
import traceback as tb
from multiprocessing import Process

import imports.utils as utils
from imports.data import MaskGenerator
from imports.data.loaders import ImageLoader


def predict(jobdir, image_dir):
    """Predict masks from images in provided directory

    :param jobdir: directory with job (settings and weights are necessary
    :param image_dir: directory with images to predict
    :return:
    """

    parser = utils.SettingsParser(os.path.join(jobdir, 'settings.json'), predict_mode=True)
    loader_args = {'load_gray': parser.loader_args['load_gray']} if 'load_gray' in parser.loader_args else {}
    loader = ImageLoader(image_dir, **loader_args)
    test = MaskGenerator(loader.get_indices(), loader, parser.batch_size, parser.aug_all, shuffle=False)

    model = parser.get_model_method()(parser.input_shape, **parser.model_params)
    model.compile(**parser.model_compile)
    model.summary()
    model.load_weights(os.path.join(jobdir, 'weights.h5'))

    pred_dir = os.path.join(jobdir, 'predicted_' + image_dir.replace('/', '_'))
    pred = model.predict_generator(test, **parser.training)
    loader.save_predicted(pred_dir, loader.get_indices(), pred)


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
