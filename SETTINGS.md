# List of necessary and optional settings
? marks optional params


- **data**
    - *images* - path with images
    - *masks* - path to ground truth with masks
    - *reg* - path to folder with registration masks (will be created if needed)
    - *descriptor* - path to file with image descriptors
- ? **registration**
    - ? *num_images* - number of nearest images for mask registration
    - ? *n_jobs* - number of threads for elastix
- **loader_type** - Usual or with registration masks - *[norm,reg]*
- ? **loader**
    - ? *train_val_test* - array of 3 values with data splits, Default - (0.8, 0.1, 0.1)
    - ? *shuffle* - shuffle data before train val test split. Default - False
    - ? [reg] *delete_previous* - Removes existing folder with registration. Default - False
- **aug_all** - List of augmentations for all data (like scaling and normalizing)
- **aug_train** - List of augmentations for train data (is applied before *aug_all*)

    Augmentations format is list of objects where *name* is keyword for augmentation and 
    then it keyword parameters follows. See next section for all possible params 

- **model** - Neural Network model
    - *name* - name of model. *One of [unet]*
        - *n_filters* - Number of filters on first level. Multiplies by two on every subsequent level. Default - 16
        - *dropout* - Dropout rate. Default - 0.5
        - *batchnorm* - Batch normalization. Default - True
        - *kernel_size* - Size of conv kernel. Default - 3
        - *activation* - Activation function. Default - 'relu'
        - *n_conv_layers* - Number of convolutional layers on every layer. Default - 2
- **model_compile** - params for model compilation
    - *optimizer* - chosen optimizer
    - ? *loss* - chosen loss
    - ? *metrics* - array of used metrics
- **training** - params for model training
    - *batch_size* - default 1
    - *epochs*
    - ? *callbacks* - array of callbacks for training.
    - ? *use_multiprocessing* - Use multiprocessing to load data from generators
    - ? *workers* - number of worker threads(processes) to load data from generators
    Callback have to be from this array *[early_stop, tensorboard, checkpoint, keep_settings]*.
    Callbacks monitors first metric on validation
- ? **predict** - Save predicted test set

## Augmentations
**All**:
- ToFloat - {name: *float*, ? max_value, ? p}

**Train**: