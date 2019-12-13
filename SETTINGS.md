# List of necessary and optional settings
? marks optional params


- **data**
    - *images* - path with images
    - *masks* - path to ground truth with masks
    - *reg* - path to folder with registration masks (will be created if needed)
    - *descriptor* - path to file with image descriptors
    - *input_shape* - input shape of data. It's up to user to determine valid shape
- ? **registration**
    - ? *num_images* - number of nearest images for mask registration
    - ? *n_jobs* - number of threads for elastix
- **loader_type** - Usual or with registration masks - *[norm,reg]*
- ? **loader_decorators** - decorators for loader
- ? **loader**
    - ? *train_val_test* - array of 3 values with data splits, Default - (0.8, 0.1, 0.1)
    - ? *shuffle* - shuffle data before train val test split. Default - False
    - ? *load_gray* - loads grayscale image if True, RGB if False
    - ? *mask_channel_codes* - int (will become [0 ... val - 1] or array. Default - None for one channel mask
    - ? [reg] *delete_previous* - Removes existing folder with registration. Default - False
- **aug_all** - List of augmentations for all data (like scaling and normalizing)
- **aug_train** - List of augmentations for train data (is applied before *aug_all*)

    Augmentations format is list of objects where *name* is keyword for augmentation and 
    then it keyword parameters follows. See next section for all possible params 

- **model** - Neural Network model
    - *name* - name of model. *One of [unet]*
        - *out_channels* - Number of output channels
        - *n_filters* - Number of filters on first level. Multiplies by two on every subsequent level. Default - 16
        - *dropout* - Dropout rate. Default - 0.5
        - *batchnorm* - Batch normalization. Default - True
        - *kernel_size* - Size of conv kernel. Default - 3
        - *activation* - Activation function. Default - 'relu'
        - *n_conv_layers* - Number of convolutional layers on every layer. Default - 2
- **model_compile** - params for model compilation
    - *optimizer* - chosen optimizer
    - ? *loss* - chosen loss. One of *['jaccard','dice','binary_focal']* 
    or any of keras builtin losses (such as 'binary_crossentropy')
    - ? *metrics* - array of used metrics. Metrics have to be keras builtins or from *['iou','f1','f2',
    'precision','recall']*
- **training** - params for model training
    - *batch_size* - default 1
    - *epochs*
    - ? *callbacks* - array of callbacks for training.
    - ? *use_multiprocessing* - Use multiprocessing to load data from generators
    - ? *workers* - number of worker threads(processes) to load data from generators
    Callback have to be from this array *[early_stop, tensorboard, checkpoint, keep_settings]*.
    Callbacks monitors first metric on validation
- ? **predict** - Save predicted test set
- ? **show_sample** - Show train sample to check augmentation

## Augmentations
Argument *p* - probability of applying can be used in all augmentations

**All**:
- **Pixel**
    - Equalize - {name: *equalize*, ?mode, ?by_channels}
    - ToFloat - {name: *float*, ?max_value}
    - CLAHE - {name: *clahe*, ?clip_limit, ?tile_grid_size}
- **Spatial**
    - Resize {name: *resize*, height, width, ?interpolation}

**Train**:
- **Pixel**
    - Blur - {name: *blur*, ?blur_limit}
    - Downscale - {name: *downscale*, ? scale_min, ?scale_max}
    - GaussNoise {name: *gauss_noise*, ?var_limit, ?mean}
    - GaussianBlur {name: *gauss_blur*, ?blur_limit}
    - IAASharpen {name: *iaa_sharpen*, ?alpha, ?lightness}
    - ImageCompression {name: *compression*, ?quality_lower, ?quality_upper}
    - ISONoise {name: *iso_noise*, ?color_shift, ?intensity}
    - MedianBlur {name: *median_blur*, ?blur_limit}
    - RandomBrightnessContrast - {name: *bright_contrast*, ?brightness_limit, ?contrast_limit, ?brightness_by_max}
- **Spatial**
    - ShiftScaleRotate - {name: *shift_scale_rotate*, ?shift_limit, ?scale_limit, ? rotate_limit)
    - RandomCrop - {name: *crop*, height, width}
    - RandomSizedCrop - {name: *sized_crop*, min_max_height, height, width}

## Decorators
**Loader**
- Ignore label - {name: *ignore_label*, ?radius, ?channel}