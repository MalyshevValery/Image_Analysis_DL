"""Maps settings strings to functions and objects"""
import segmentation_models.losses as losses
import segmentation_models.metrics as metrics

# import imports.data.decorators as decorators

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