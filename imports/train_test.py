import os
import sys
import numpy as np

from imports.utils.gpu_setup import gpu_setup
from imports.utils.settings_parser import SettingsParser
from imports.data.mask_generator import MaskGenerator
import matplotlib.pyplot as plt
import json


def train_test(settings_filename='settings.json'):
    with open(os.path.join(os.path.dirname(__file__), '..', 'gpu_settings.json'), 'r') as gpu_file:
        gpu_settings = json.load(gpu_file)
        gpu_setup(gpu_settings)

    parser = SettingsParser(settings_filename)
    loader = parser.get_loader()
    train = MaskGenerator(loader.train_indices(), loader, parser.batch_size, parser.aug_train)
    val = MaskGenerator(loader.valid_indices(), loader, parser.batch_size, parser.aug_all)
    test = MaskGenerator(loader.test_indices(), loader, parser.batch_size, parser.aug_all, shuffle=False)

    if parser.show_sample:
        to_show = train[0]
        image = to_show[0][0]
        mask = to_show[1][0]
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        image[:, :, :mask.shape[2]] *= (1 - mask)
        plt.imshow(image)
        plt.show()

    model = parser.get_model_method()(parser.input_shape, **parser.model_params)
    model.compile(**parser.model_compile)
    model.summary()

    callbacks = parser.get_callbacks()
    results = model.fit_generator(train, epochs=parser.epochs, validation_data=val, callbacks=callbacks,
                                  **parser.training)
    model.load_weights(os.path.join(parser.results_dir, 'weights.h5'))

    ret = model.evaluate_generator(test, callbacks=callbacks, **parser.training)
    ret_val = {'loss': ret[0]}
    if len(ret) > 1:
        for i, n in enumerate(parser.metrics_names):
            ret_val[n] = ret[i + 1]
    print('Test resutls: ', ret_val)
    for key in ret_val:
        open(os.path.join(parser.results_dir, '%.2f_' % ret_val[key] + key), 'w')

    if parser.predict:
        pred_dir = os.path.join(parser.results_dir, 'predicted')
        pred = model.predict_generator(test, callbacks=callbacks, **parser.training)
        loader.save_predicted(pred_dir, loader.test_indices(), pred)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_test()
    else:
        train_test(sys.argv[1])
