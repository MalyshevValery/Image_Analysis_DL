"""Main training script"""
import argparse
import os
import sys
import traceback as tb
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np

import imports.utils as utils
from imports.data import MaskGenerator
from imports.data.storages import DirectoryStorage


def train(settings='settings.json', show_sample=False, predict=False):
    """Train model

    :param settings: path to settings file
    :param show_sample: shows sample from input data if True
    :param predict: predicts and saves Test set if True
    """

    parser = utils.SettingsParser(settings)
    loader = parser.get_loader()
    js = loader.to_json()
    print(js)
    loader = loader.from_json(js)
    train_keys, val_keys, test_keys = loader.split(parser.settings['train_val_test'])
    train_gen = MaskGenerator(train_keys, loader, parser.batch_size, parser.augmentation_train_merged)
    val_gen = MaskGenerator(val_keys, loader, parser.batch_size, parser.augmentation_all)
    test_gen = MaskGenerator(test_keys, loader, parser.batch_size, parser.augmentation_all, shuffle=False)

    if show_sample:
        to_show = train_gen[0]
        image = to_show[0][0]
        mask = to_show[1][0]
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        image[:, :, :mask.shape[2]] *= (1 - mask)
        plt.imshow(image)
        plt.show()

    model = parser.get_model_method()(parser.settings['input_shape'], **parser.model_params)
    model.build((None, *parser.settings['input_shape']))
    model.compile(**parser.model_compile)
    model.summary()

    callbacks = parser.get_callbacks()
    model.fit_generator(train_gen, epochs=parser.epochs, validation_data=val_gen, callbacks=callbacks,
                        **parser.training)
    model.load_weights(os.path.join(parser.results_dir, 'weights.h5'))

    ret = model.evaluate_generator(test_gen, callbacks=callbacks, **parser.training)
    ret_val = {'loss': ret[0]}
    if len(ret) > 1:
        for i, n in enumerate([name for name in parser.metrics_names if name != 'loss']):
            ret_val[n] = ret[i + 1]
    print('Test results: ', ret_val)
    for key in ret_val:
        open(os.path.join(parser.results_dir, '%.3f_' % ret_val[key] + key), 'w')

    if predict:
        pred_dir = os.path.join(parser.results_dir, 'predicted')
        pred = model.predict_generator(test_gen, callbacks=callbacks, **parser.training)
        loader.save_predicted(test_keys, pred, DirectoryStorage(pred_dir, mode='w'))


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
