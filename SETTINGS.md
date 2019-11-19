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
- **generator_type** - Usual or with registration masks - *[norm,reg]*
- ? **generator**
    - ? *train_val_test* - array of 3 values with data splits
    - ? *shuffle* - shuffle data before train val test split
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