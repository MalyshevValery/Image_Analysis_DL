# XRay_Segmentation
Project for developing Neural Networks for XRay

## Prerequisites
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
`run/` directory includes executable scripts
In order to run scripts python virtual environment must be activated `. venv/bin/activate`

Executable scripts:
1. **check.py** - takes JSON configuration or folder with some of them and validate provided configuration
2. **train.py** - script which is used to train models defined by one or some JSON configurations
3. **predict.py** - this scripts takes directory with job or with other job directories and directory
with images. For every job directory the script loads configuration and weights and perform
inference for models
4. **convert.py** - this script takes special configuration and perform data conversion from loader
to described storages. You may need it if you want to speed up data loading or save already 
preprocessed data

## Settings
Detailed explanation of available settings is listed in *SETTINGS.md*. 
Example of settings is listed in *settings.json*.