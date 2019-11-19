import os
import sys
from imports.settings_parser import SettingsParser


def train_test(settings_filename='settings.json'):
    parser = SettingsParser(settings_filename)
    generator, img_shape = parser.get_data_generator()

    model = parser.get_model_method()(img_shape, **parser.model_params)
    model.compile(**parser.model_compile)
    model.summary()

    callbacks = parser.get_callbacks()
    generator.set_batch_size(parser.batch_size)
    results = model.fit_generator(generator.train_generator(), epochs=parser.epochs,
                                  steps_per_epoch=generator.train_steps(), validation_data=generator.valid_generator(),
                                  validation_steps=generator.valid_steps(), callbacks=callbacks, **parser.training)

    ret = model.evaluate_generator(generator.test_generator(), generator.test_steps(), callbacks=callbacks,
                                   **parser.training)
    ret_val = {'loss': ret[0]}
    if len(ret) > 1:
        for i, n in enumerate(parser.metrics_names):
            ret_val[n] = ret[i + 1]
    print('Test resutls: ', ret_val)
    model.save('Models/' + parser.general_name + '-final.h5')
    open('Models/test_' + str(ret_val) + "_" + parser.general_name, 'w')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_test()
    else:
        train_test(sys.argv[1])
