import unittest

from imports.utils import RunParams, DEFAULT_PARAMS


class RunParamsTester(unittest.TestCase):
    def test_init(self) -> None:
        rp = RunParams(epochs=5, verbose=False, shuffle=True,
                       class_weight={1: 0.5}, workers=5,
                       use_multiprocessing=True)
        self.assertDictEqual(rp.dict, {
            'epochs': 5,
            'verbose': False,
            'shuffle': True,
            'initial_epoch': 0,
            'class_weight': {1: 0.5},
            'workers': 5,
            'use_multiprocessing': True,
            'max_queue_size': 10,
            'validation_freq': 1,
            'sample_weight': None
        })

    def test_eval(self) -> None:
        eval_dict = DEFAULT_PARAMS.eval_dict
        self.assertSetEqual(set(eval_dict.keys()), {
            'verbose', 'workers', 'use_multiprocessing', 'max_queue_size',
        })
