from typing import Tuple

import abc
from collections import namedtuple

import tensorflow as tf

from utils.etc_utils import isnamedtupleinstance
from data.vocabulary import Vocabulary

_scalar = lambda: tf.TensorShape([])
_vector = lambda: tf.TensorShape([None])
_matrix = lambda: tf.TensorShape([None, None])
_tensor = lambda: tf.TensorShape([None, None, None])


class DatasetReader(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read(self,
             file_path: str,
             mirrored_strategy: tf.distribute.Strategy = None) -> Tuple[tf.data.Dataset, int]:
        pass

    @property
    @abc.abstractmethod
    def iterator_shapes(self):
        pass

    @property
    @abc.abstractmethod
    def iterator_types(self):
        pass

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary  # pylint: disable=no-member


def string_split(line, dtype=tf.int32):
    return tf.compat.v1.string_to_number(tf.compat.v1.string_split([line]).values, dtype)


def list_of_string_split(line,
                       dtype=tf.int32,
                       delimiter: str = ' ',
                       padding="0"):
    splitted_sentences = tf.sparse.to_dense(tf.compat.v1.string_split(line, sep=delimiter), default_value=padding)
    splitted_sentences = tf.compat.v1.string_to_number(splitted_sentences, dtype)
    sentence_lengths = tf.compat.v1.count_nonzero(splitted_sentences, axis=1)
    return splitted_sentences, sentence_lengths


def list_of_list_of_string_split(line,
                                 dtype=tf.int32,
                                 first_delimiter: str = '\n',
                                 second_delimiter: str = ' ',
                                 padding="0"):
    # First split
    splitted_sentences = tf.sparse.to_dense(tf.compat.v1.string_split(line, sep=first_delimiter), default_value=padding)
    shape = tf.shape(splitted_sentences)
    flattened_sentences = tf.reshape(splitted_sentences, [-1])

    # Second split
    splitted_sentences = tf.sparse.to_dense(tf.compat.v1.string_split(flattened_sentences, sep=second_delimiter), default_value=padding)
    splitted_sentences = tf.compat.v1.string_to_number(splitted_sentences, dtype)
    splitted_sentences = tf.reshape(splitted_sentences, [shape[0], shape[1], -1])

    sentence_lengths = tf.compat.v1.count_nonzero(splitted_sentences, axis=2)
    num_sentences = tf.compat.v1.count_nonzero(sentence_lengths, axis=1)

    return splitted_sentences, sentence_lengths, num_sentences


def bucketing(dataset, key_name, batch_size, bucket_width, mode, padded_shapes=None):
    # TODO: If we use multi-gpu, we need to set same bucket across gpus
    # Maybe we can just split batch by number of gpus
    # https://stackoverflow.com/a/46966248

    def _batching_func(x, drop_remainder=False):
        return x.padded_batch(batch_size, padded_shapes=padded_shapes,
                              drop_remainder=drop_remainder)

    if 'train' in mode:
        def _key_func(data):
            if isnamedtupleinstance(data):
                bucket_key = getattr(data, key_name)
            elif isinstance(data, dict):
                bucket_key = data[key_name]
            else:
                raise TypeError("Must be namedtuple or dict")
            bucket_id = bucket_key / bucket_width
            return tf.cast(bucket_id, tf.int64)
            # return tf.to_int64(bucket_id)

        def _reduce_func(unused_key, windowed_data):
            return _batching_func(windowed_data)

        batched_dataset = dataset.apply(
            tf.data.experimental.group_by_window(
                key_func=_key_func,
                reduce_func=_reduce_func,
                window_size=batch_size
            )
        )
    elif 'test' in mode or 'valid' in mode:
        batched_dataset = _batching_func(dataset, True)

    return batched_dataset


def tensor_pad(x, max_lengths):
    for i, max_length in enumerate(max_lengths):
        shape = []
        for j, dim in enumerate(tf.unstack(tf.shape(x))):
            if i == j:
                shape.append(max_length - dim)
            else:
                shape.append(dim)
        shape = tf.stack(shape)
        padding = tf.zeros(shape, dtype=x.dtype)
        x = tf.concat([x, padding], axis=i)
    return x
