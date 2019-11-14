import os
from imports.settings_parser import SettingsParser

if __name__ == '__main__':
    parser = SettingsParser(os.path.join(os.path.dirname(__file__), 'settings.json'))
    generator = parser.get_data_generator()

    img_shape = (256, 256, 3)
    model = parser.get_model_method()(img_shape, **parser.model_params)
    model.compile(**parser.model_compile)
    model.summary()

    generator.set_batch_size(parser.batch_size)
    results = model.fit_generator(generator.train_generator(), epochs=parser.epochs,
                                  steps_per_epoch=generator.train_steps(), validation_data=generator.valid_generator(),
                                  validation_steps=generator.valid_steps(), callbacks=parser.get_callbacks())

    model.save('Models/' + parser.general_name + '-final.h5')
    parser.keep_settings()
