import segmentation_models.metrics as metrics
import segmentation_models.losses as losses
import imports.data.decorators as decorators
import albumentations as aug
from imports.data.loaders import ImageMaskLoader, ImageRegMaskLoader

metrics_map = {
    'iou': metrics.IOUScore(name='iou'),
    'f1': metrics.FScore(beta=1, name='f1'),
    'f2': metrics.FScore(beta=2, name='f2'),
    'precision': metrics.Precision(name='precision'),
    'recall': metrics.Recall(name='recall')
}

loss_map = {
    'jaccard': losses.JaccardLoss(),
    'dice': losses.DiceLoss(),
    'binary_focal': losses.BinaryFocalLoss()
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
    'norm': ImageMaskLoader,
    'reg': ImageRegMaskLoader
}

decorators = {
    'ignore_label': decorators.ignore_label_loader
}
