# Settings
All settings have to be located in JSON file between `{}` as well as subgroups settings. 

For example *generator_params* have to look like `"generator_params": { "epochs": 1, "verbose": 1}`

Settings are listed in a next way: `<setting name> (<default value if exists>) - <explanation>`

**Required arguments**, *optional arguments*

### Main
*This configuration is root object of JSON file and is used by train, test and check scripts*

All values can be used for [hyperparameter optimization](#hypopt)

- **loader** - main data source
    - [**images**](#storage) - source with input images
    - [*masks*](#storage) - source of masks for semantic segmentation
    - [*extensions*](#extension) - different data preprocessing/postprocessing transformations that are applied
    before augmentation
- [**model**](#model) - deep learning model which will be created and used
- *job_dir* (auto generated from timestamp and settings filename) - name of job directory
- [*augmentation_train*](#augmentation) - specifies augmentations for training dataset
- [*augmentation_all*](#augmentation) - specifies augmentations for all datasets. For training dataset it is applied
after *augmentation_train*. Though validation and test datasets should not be augmented use this in
order to scale input data AFTER training augmentation (some of them are more effecient if are done 
before scaling)
- [*callbacks*](#callback) - list `[]` of callbacks which will be used during training process
- [*model_compile*](#compile) - parameters used for model compilation
- *train_val_test* ([0.8, 0.1, 0.1]) - tuple or list of 3 values which declare split sizes for training
validation and test datasets. Split is random
- *batch_size* (1) - batch size for the whole process of training (it is passed to generators)
- *restore_weights* (True) - if True model will try to load best weights saved to `weights.h5` by
checkpoint callback
- *generator_params*
    - *epochs* (1) - number of epochs for training
    - *verbose* (1) - level of verbosity during training process
    - *class_weight* - dictionary mapping class indices to a weight for the class.
    - *max_queue_size* (10) - maximum size of generator queue
    - *workers*: (1) - number of processes to load data from generator
    - *use_multiprocessing*: (False) - if true process-based threading will be used
    - *shuffle* (True) - shuffle generator indices
    - *initial_epoch* (0) - starting epoch
- *eval_metric* (loss) - metric which is used for evaluation test set during hyperparameter optimization


<a name="storage"></a>
### Storage
Storage is the class that defines how and from where the data is loaded or saved to. 
For example from directory, file or internet. Storages have modes - read or write and operates with
keys. So when data is passed to loader from different storages, loader keep set of keys which is
intersection of sets of keys from different storages. It means that data should be ordered with same
keys to keep track of bonds between input and output data.

Currently implemented storages:
- Directory storage - loads and saves data to directory with filenames as keys
    - **type** = "directory"
    - **dir** - directory to load or save from
    - **color_mode** ("none") - Can be one of *[none, to_gray, from_gray]*. When `to_gray` is used it 
    transforms image to one channel grayscale from input or vice versa if `from_gray` is specified.
- HDF5 storage - loads and saves data as arrays in hdf5 file.
    - **type** = "hdf5"
    - **filename** - path to hdf5 file
    - **dataset_name** - name of dataset inside specified file

<a name="extension"></a>
### Extensions
Extensions are transformation that are used inside of loader on certain point of data processing.
Currently 4 such points exists:
- image - is applied after retrieving input image data
- mask - is applied after retrieving mask form storage
- save_image - is used after loading input image in `save_predicted` method (mostly used in test
evaluation and inference mode). It is needed because augmentation is not used in inference mode so
scaling as last step in augmentation should be compensated by extension
- save - is used before saving image to storage

Structure of extensions configuration is object `{}` where keys are points of extension application
and values are lists `[]` of extensions to be used at that point. It is possible to use one extension
without list.

Now the following extensions can be used:
- TypeScale extension - allows to linearly scale data and change its type
    - **type** = "type_scale"
    - *src_max* (every item maximum if not specified) - maximum value of current type to scale from.
    If not specified data entries will be scaled according to each own maximum value.
    - *target_type* (uint8) - target type to change to. Currently available types are `uint8` and
    `float 32`
    - *target_max* (1 if target type is float, max possible value for target type in case of integer type) - maximum
    value to scale to.
- SplitMask - allows to split one channels mask on some channel by provided codes of different classes
in that single channel.
    - **type** = "split_mask"
    - *codes* (2) - codes to determine channels for new mask. If integer `n` is specified then codes are [0 ... n - 1]
    otherwise provided list is used as codes
- IgnoreRegion - creates empty region on multichannel mask to give model some degree of freedom in its
predictions. Note that it does not work very well with losses that penalize false positives. That empty
region is created by performing morphological operations on one of channels that is specified as main.
- **type** = "ignore_region"
- *radius* (3) - radius of disk structural element that is used in morphological operations
- *channel* (1) - main channel to perform morphological operations on

<a name="model"></a>
### Model
Currently available models:
- UNet - CNN for semantic segmentation
    - **name** = UNet
    - *input_shape* (is derived from shape of data sample loaded from input data if not provided) -
    shape of one sample of input data
    - *out_channels* (1) - number of output channels of model
    - *n_filters* (16) - number of filters in convolution. Number of filters is progressively increased
    with [1, 2, 4, 8, 16] multiplier with every further block of convolution layers
    - *dropout* (0.5) - dropout rate
    - *batchnorm* (True) - if true batch normalization will be applied after convolution layers
    - *kernel_size* (3) - size of convolution kernel
    - *activation* (relu) - label of activation function
    - *n_conv_layers* (2) - number of convolution layers on each block 

<a name="callback"></a>
### Callbacks
These settings specify keras callbacks which are used during training process.
**NOTE** that it is possible to use `$BASE$` when specify directories to refer to job directory.

Currently available callbacks
- Early stop - stops training process when the best metric value is acquired
    - **name** = "early_stop"
    - *monitor* (val_loss) - metric to monitor, add `val_` to metric name to monitor 
    metrics value on validation
    - *min_delta*: (0) - minimum change which will be considered as an improvement
    - *patience*: (0) - how many epochs without performance improve will pass before stopping
    - *verbose* (0) - level of verbosity
    - *mode* (auto) - One of [max, min]. This setting specifies whether high value of 
    metric is desired or low. It is recommended to not use auto mode
    - *baseline* (0) - training will stop only if monitored value is above than baseline
    - *restore_best_weights* (False) - if true weights with the best value of monitored 
    metric will be restored after stopping
- ModelCheckpoint - saves weights of model with specified frequency
    - **name** = "checkpoint"
    - **filepath** - path where save model 
    - *monitor* (val_loss) - metric to monitor, add `val_` to metric name to monitor 
    metrics value on validation
    - *verbose* (0) - level of verbosity
    - *save_best_only* (False) - if true saves only best epoch according to selected metric
    - *save_weights_only* (False) - if true saves only weights, whole model otherwise
    - *mode* (auto) - One of [max, min]. This setting specifies whether high value of 
    metric is desired or low. It is recommended to not use auto mode
    - *save_freq* ('epoch') - if integer is used then model will be saved after looking at
    specified number of samples
- SaveBest - less configurable model checkpoint for more comfortable usage. It has preset settings:
`filepath=$BASE$/best_model.h5, save_best_only=True, save_weights_only=False`. All other settings
can be set as specified before
    - **name** = "save_best"
    - ... same as Model except the ones in preset
- Tensorboard - tensorboard callback to create logs that can be monitored by `tensorboard`
    - *log_dir* ($BASE$) - path to save logs
    - *histogram_freq* - (0) - frequency (in epochs) at which to compute activation and 
    weight histograms for the layers of the model. If set to 0, histograms won't be computed.
    - *write_graph* (True) - true to write graph of model
    - *write_images* (False) - whether to write model weights to visualize as image in TensorBoard
    - *update_freq* ('epoch') - on of ['batch', 'epoch', integer], integer specifies number of samples between each update
    - *profile_batch* (2) - profile batch to compute characteristics. **HIGHLY RECOMMENDED TO SET IT AS 0** 
    Cause this feature works only in eager execution mode of tensorflow
    - *embeddings_freq* (0) - frequency (in epochs) at which embedding layers will be visualized. If set to 0, embeddings won't be visualized

<a name="compile"></a>
### Compile
Model compilation parameters:
- *optimizer* (rmsprop) - String (name of optimizer) or optimizer instance. See `tf.keras.optimizers`.,
- *loss* - String (name of objective function) from `tf.losses` or one of `binary_focal, categorical_focal, dice, jaccard`
- *metrics* -  ist of metrics to be evaluated by the model during training and testing 
(in addition to tensorflow metrics also `iou, f1, f2, precision, recall` are provided)

<a name="augmentation"></a>
### Augmentation
To specify augmentation settings you should consult with [albumentations library documentation](https://albumentations.readthedocs.io/en/latest/api/augmentations.html)

Required augmentations can be listed as in `[]` then Compose augmentation will be created
or just one augmentation as it is. Structure of settings for one augmentation looks like `{ "name": ToFloat, <other params>}`
Parameter `"name"` specifies type of created augmentation and is the same as base class name from documentation.
Other parameters are optional parameters identical to the ones in documentation.

Example:
`class albumentations.augmentations.transforms.Blur(blur_limit=7, p=0.5)` =>
```
{
 "name": "Blur",
 "blur_limit": 7,
 "p": 0.5
}
```

<a name="hypopt"></a>
### Hyperparameter optimization
For every value in config parameter optimization can be used. To apply it instead of
specifying value add additional JSON object which specifies range of check:
- **type** - Type of value distribution, one of [O_CHOICE, O_RANDINT, O_UNIFORM, O_QUNIFORM, O_LOG_UNIFORM, O_QLOG_UNIFORM, O_UNIFORM_INT, O_NORMAL, O_QNORMAL, O_LOG_NORMAL, O_QLOG_NORMAL].
For *CHOICE* add *options* list, for *UNIFORM*-like *low* and *high* parameters are required. For *RANDINT* only low is needed.
For *NORMAL*-like *mu*(mean) and *sigma*(std). Prefix *Q* adds quantization to distribution and parameter *q* is needed for that.
*LOG* means that logarithm of value will be uniformly or normally distributed
- *low* - *UNIFORM*-like - specifies lowest value
- *high* - *UNIFORM*-like and *RANDINT* - specifies max value. For *RANDINT* interval will be `[0, high)`
- *options* - list of possible options for *CHOICE*
- *mu* - mean for *NORMAL*-like
- *sigma* - std for *NORMAL*-like
- *q* - for *Q* prefix types it means `round(<distribution>/q)*q` what adds quantization

More information in `imports/optim.py` and in [repo of hyperopt](https://github.com/hyperopt/hyperopt)