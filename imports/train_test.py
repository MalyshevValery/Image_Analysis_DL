import os
import sys

from imports.settings_parser import SettingsParser
from imports.data.mask_generator import MaskGenerator

import matplotlib.pyplot as plt

def train_test(settings_filename='settings.json'):
    parser = SettingsParser(settings_filename)
    loader, img_shape = parser.get_loader()
    train = MaskGenerator(loader.train_indices(), loader, parser.batch_size, parser.aug_all)
    val = MaskGenerator(loader.train_indices(), loader, parser.batch_size, parser.aug_all)
    test = MaskGenerator(loader.train_indices(), loader, parser.batch_size, parser.aug_all)

    print(train[0][0].max())
    plt.imshow(train[0][0][0])
    plt.show()
    plt.hist(train[0][0].flatten())
    plt.show()

    model = parser.get_model_method()(img_shape, **parser.model_params)
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
