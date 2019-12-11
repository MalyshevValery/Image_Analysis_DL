# XRay_Segmentation

## Prerequisites
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
[change tensorflow to tensorflow-gpu in requirements.txt if you want GPU version]
pip install -r requirements.txt

apt install elastix
```

## Training model

All parameters for training segmentation are located in `settings.json`.
Explanation of settings is located in `imports/settings_parser.py` docs.

To start training process adjust settings and simply run `python3 main.py`