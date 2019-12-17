"""Maps settings strings to functions and objects"""
import albumentations as aug

import imports.data.decorators as decorators
import imports.data.loaders as loaders

metrics_map = {
}

loss_map = {
}

augmentations = {
    'blur': aug.Blur,
    'bright_contrast': aug.RandomBrightnessContrast,
    'clahe': aug.CLAHE,
    'crop': aug.RandomCrop,
    'sized_crop': aug.RandomSizedCrop,
    'compression': aug.ImageCompression,
    'downscale': aug.Downscale,
    'equalize': aug.Equalize,
    'float': aug.ToFloat,
    'gauss_noise': aug.GaussNoise,
    'gauss_blur': aug.GaussianBlur,
    'iaa_sharpen': aug.IAASharpen,
    'iso_noise': aug.ISONoise,
    'median_blur': aug.MedianBlur,
    'resize': aug.Resize,
    'shift_scale_rotate': aug.ShiftScaleRotate
}

loader_class = {
    'norm': loaders.ImageMaskLoader,
    'reg': loaders.ImageRegMaskLoader
}

decorators = {
    'ignore_label': decorators.add_ignore
}
