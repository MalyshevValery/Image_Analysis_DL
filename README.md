# XRay_Segmentation

## Prerequisites
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install <tensorflow==2.0.0b1 / tensorflow-gpu==2.0.0b1>
```

## Training model

All parameters for training segmentation are located in `settings.json`.
Explanation of settings is located in `imports/settings_parser.py` docs.

To start training process adjust settings and simply run `python3 main.py`