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


def train(settings_filename='settings.json'):
    """Train model

    :param settings_filename: path to settings file
    """

    parser = utils.SettingsParser(settings_filename)
    loader = parser.get_loader()
    train_gen = MaskGenerator(loader.train_indices(), loader, parser.batch_size, parser.aug_train)
    val_gen = MaskGenerator(loader.valid_indices(), loader, parser.batch_size, parser.aug_all)
    test_gen = MaskGenerator(loader.test_indices(), loader, parser.batch_size, parser.aug_all, shuffle=False)

    if parser.show_sample:
        to_show = train_gen[0]
        image = to_show[0][0]
        mask = to_show[1][0]
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        image[:, :, :mask.shape[2]] *= (1 - mask)
        plt.imshow(image)
        plt.show()

    model = parser.get_model_method()(parser.input_shape, **parser.model_params)
    model.build((None, *parser.input_shape))
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

    if parser.predict:
        pred_dir = os.path.join(parser.results_dir, 'predicted')
        pred = model.predict_generator(test_gen, callbacks=callbacks, **parser.training)
        loader.save_predicted(pred_dir, loader.test_indices(), pred)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('settings', help='File with settings or directory with different settings', nargs='?',
                      default='settings.json')
    parsed_args = args.parse_args(sys.argv[1:])

    settings = parsed_args.settings
    if os.path.isfile(settings):
        print('File -', settings)
        train(settings)
    elif os.path.isdir(settings):
        print('Dir -', settings)
        for f in os.listdir(settings):
            print('File -', os.path.join(settings, f))
            try:
                proc = Process(target=train, args=(os.path.join(settings, f),))
                proc.start()
                proc.join()
            except Exception as e:
                tb.print_exc()
