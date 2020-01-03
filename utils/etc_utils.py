import os
import logging

# for set_logger
import logging
import colorlog

# for sort_dict
import operator

# for timestamp
import calendar
import datetime

# for split_list
import more_itertools

# for load_json
import json
import random

import tensorflow as tf

NEAR_INF = 1e20


def set_logger(fname=None):
    colorlog.basicConfig(
        filename=fname,
        level=logging.INFO,
        format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def timestamp_to_utc(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp)


def utc_to_timestamp(utc):
    return calendar.timegm(utc.utctimetuple())


def split_list(in_list, num_splits):
    return [list(c) for c in more_itertools.divide(num_splits, in_list)]


def sort_dict(dic):
    # Sort by alphabet
    sorted_pair_list = sorted(dic.items(), key=operator.itemgetter(0))
    # Sort by count
    sorted_pair_list = sorted(sorted_pair_list, key=operator.itemgetter(1), reverse=True)
    return sorted_pair_list


def load_json(fname):
    colorlog.info(f"Read {fname}")
    jsons = []
    with open(fname, 'r') as fp:
        for line in fp:
            jsons.append(json.loads(line))
    return jsons


def set_tcmalloc(path="/usr/lib/libtcmalloc.so"):
    if os.path.exists(path):
        os.environ["LD_PRELOAD"] = path
    else:
        logging.warning(f"{path} not exists. There might be performance loss" \
                        " when you use data parallelism")


def set_gpus(gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n)==str for n in f)


def check_none_gradients(grads, vars, ignore_none=False):
    is_none = False
    for grad, var in zip(grads, vars):
        if grad is None:
            colorlog.error(f"{var.name} has None gradient!")
            is_none = True
    if is_none and not ignore_none:
        from IPython import embed; embed()  # XXX DEBUG


def check_nan_gradients(grads, vars):
    is_nan = False
    nan_vars = []
    nan_grads = []
    for grad, var in zip(grads, vars):
        if isinstance(grad, tf.IndexedSlices):
            grad = grad._values

        if tf.math.is_nan(tf.reduce_sum(grad)):
            is_nan = True
            nan_vars.append(var.name)
            nan_grads.append(grad)

    if is_nan:
        import pudb; pudb.set_trace()  # XXX DEBUG
