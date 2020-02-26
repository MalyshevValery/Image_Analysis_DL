"""Class to store and convert params for Model.fit and Model.evaluate"""
from typing import NamedTuple, Dict, Optional

import numpy as np


class RunParams(NamedTuple):
    """Class to store simple params for Model.fir and Model.evaluate

    Callbacks is not stored here as well as validation params (validation params
    are reflected in Loader after train_test_split

    - epochs (1)
    - initial_epoch (0)
    - validation_freq (1)

    - verbose (True)
    - shuffle (True)

    - class_weight (None)
    - sample_weight (None)

    - max_queue_size (10)
    - workers (1)
    - use_multiprocessing (False)
    """
    epochs: int = 1
    verbose: bool = True
    shuffle: bool = True
    class_weight: Optional[Dict[int, float]] = None
    sample_weight: np.ndarray = None
    initial_epoch: int = 0
    validation_freq: int = 1
    max_queue_size: int = 10
    workers: int = 1
    use_multiprocessing: bool = False

    @property
    def dict(self) -> Dict[str, object]:
        """Returns parameters as dictionary"""
        return self._asdict()

    @property
    def eval_dict(self) -> Dict[str, object]:
        """Returns dict with params suitable for evaluation"""
        params = self.dict
        params.pop('initial_epoch')
        params.pop('epochs')
        params.pop('validation_freq')
        params.pop('shuffle')
        params.pop('class_weight')
        params.pop('sample_weight')
        return params


DEFAULT_PARAMS = RunParams()
