"""Wrapper for hyperparameter optimization that can be used during training"""
from collections import Iterable
from copy import deepcopy

import hyperopt
import hyperopt.hp as hp


def __kwargs_to_args(fun):
    def wrapper(label, low=None, high=None, options=None, mu=None, sigma=None, q=None):
        """Fast wrapper to translate kwargs to args

        :param label: label for option
        :param low: low for UNIFORM like
        :param high: high for UNIFORM like and RANDING
        :param options: only for CHOICE
        :param mu: mean for NORMAL like
        :param sigma: std for NORMAL like
        :param q: q value for Q... like -> discrete values (step)
        :return: wrapper for proper function
        """
        args = [label]
        for option in [low, high, options, mu, sigma, q]:
            if option is not None:
                args.append(option)
        return fun(*args)

    return wrapper


# Map to translate JSON names to Hyperopt
__NAME_TO_HYPER_MAP = {
    'O_CHOICE': __kwargs_to_args(hp.choice),  # choice among list
    'O_RANDINT': __kwargs_to_args(hp.randint),  # random int in [0, high) range

    'O_UNIFORM': __kwargs_to_args(hp.uniform),  # uniform distribution between low and high
    'O_QUNIFORM': __kwargs_to_args(hp.quniform),  # round(uniform(low, high) / q) * q
    'O_LOG_UNIFORM': __kwargs_to_args(hp.loguniform),  # exp(uniform(low, high))
    'O_QLOG_UNIFORM': __kwargs_to_args(hp.qloguniform),  # round(exp(uniform(low, high)) / q) * q
    'O_UNIFORM_INT': __kwargs_to_args(hp.uniformint),  # integer uniform

    'O_NORMAL': __kwargs_to_args(hp.normal),  # normally distributed with mean mu and std sigma
    'O_QNORMAL': __kwargs_to_args(hp.qnormal),  # round(normal(mu, sigma) / q) * q
    'O_LOG_NORMAL': __kwargs_to_args(hp.lognormal),  # exp(normal(mu, sigma))
    'O_QLOG_NORMAL': __kwargs_to_args(hp.qlognormal)  # round(exp(normal(mu, sigma)) / q) * q
}


def get_space_for_json(json):
    """Returns hyperopt space for JSON and changes according values from it to a placeholder"""
    space = {}
    __process_json(json, accumulated_space=space)
    return space


def optimize(fun, json, space, trials=10):
    """Starts running trials

    :param fun: function to optimize
    :param json: config after get_space_for_json()
    :param space: acquired search space
    :param trials: number of trials for selecting best parameter
    :return:
    """
    best = hyperopt.fmin(__get_fun_with_json(fun, json), space, hyperopt.tpe.suggest, trials, show_progressbar=False)
    return best


def fill_json_with_params(json, params):
    """Fills hyper params with values from space"""
    copy = deepcopy(json)
    for key in params:
        path = key.split(':')
        cur = copy
        for p in path[:-1]:
            p = p if isinstance(cur, dict) else int(p)
            cur = cur[p]
        p = path[-1]
        p = p if isinstance(cur, dict) else int(p)
        cur[p] = params[key]
    return copy


def __params_to_str(params: dict):
    key_list = list(params.keys())
    key_list.sort()
    add_string = ''
    for key in key_list:
        val = params[key]
        key = key.split(':')[-1]
        if isinstance(val, float):
            val = round(val, 2)
        add_string += key + '_' + str(val) + '_'
    return add_string


def __process_json(json, accumulated_space: dict, path_string=''):
    for key in (json if isinstance(json, dict) else range(len(json))):
        if not isinstance(json[key], Iterable) or isinstance(json[key], str):
            continue
        tmp_path_string = path_string + (':' if path_string != '' else '') + str(key)
        json[key] = __check_and_add_param(json[key], accumulated_space, tmp_path_string)
        if not isinstance(json[key], str):
            __process_json(json[key], accumulated_space, tmp_path_string)
    return json


def __check_and_add_param(json, accumulated_space: dict, path_string=''):
    if 'type' not in json:
        return json
    if json['type'] not in __NAME_TO_HYPER_MAP:
        return json

    fun = __NAME_TO_HYPER_MAP[json['type']]
    del json['type']
    accumulated_space[path_string] = fun(path_string, **json)
    return '###' + path_string + '###'


def __get_fun_with_json(fun, json):
    """Returns function takes space args as input"""
    return lambda space_params: fun(fill_json_with_params(json, space_params), __params_to_str(space_params))
