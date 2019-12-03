import os
import sys

from imports.data.loaders.image_loader import ImageLoader
from imports.settings_parser import SettingsParser
from imports.data.mask_generator import MaskGenerator
import argparse


def predict(jobdir, image_dir, frac=1):
    parser = SettingsParser(os.path.join(jobdir, 'settings.json'))
    loader_args = {}
    if 'load_gray' in parser.loader_args:
        loader_args['load_gray'] = parser.loader_args['load_gray']
    if 'mask_channel_codes' in parser.loader_args:
        loader_args['mask_channel_codes'] = parser.loader_args['mask_channel_codes']
    loader = ImageLoader(image_dir, frac, **loader_args)
    test = MaskGenerator(loader.test_indices(), loader, parser.batch_size, parser.aug_all, shuffle=False)

    model = parser.get_model_method()(parser.input_shape, **parser.model_params)
    model.compile(**parser.model_compile)
    model.summary()
    model.load_weights(os.path.join(jobdir, 'weights.h5'))

    pred_dir = os.path.join(jobdir, 'predicted_' + image_dir.replace('/', '_'))
    pred = model.predict_generator(test, **parser.training)
    loader.save_predicted(pred_dir, loader.test_indices(), pred)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('jobdir', help='Directory with trained job to load and predict')
    args.add_argument('image_dir', help='Directory with images to predict')
    args.add_argument('-f', type=float, help='Fracture of images that will be used as input', default=1)
    parsed_args = args.parse_args(sys.argv[1:])
    predict(parsed_args.jobdir, parsed_args.image_dir, parsed_args.f)
