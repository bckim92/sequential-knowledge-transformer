import os
import pickle
from typing import Dict

import colorlog
import tensorflow as tf
import numpy as np

__PATH__ = os.path.abspath(os.path.dirname(__file__))

_PARLAI_PAD = '__null__'
_PARLAI_GO = '__start__'
_PARLAI_EOS = '__end__'
_PARLAI_UNK = '__unk__'
_PARLAI_START_VOCAB = [_PARLAI_PAD, _PARLAI_GO, _PARLAI_EOS, _PARLAI_UNK]
PARLAI_PAD_ID = 0
PARLAI_GO_ID = 1
PARLAI_EOS_ID = 2
PARLAI_UNK_ID = 3

_BERT_PAD = "[PAD]"
_BERT_UNK = "[UNK]"
_BERT_CLS = "[CLS]"
_BERT_SEP = "[SEP]"
_BERT_MASK = "[MASK]"
BERT_PAD_ID = 0
BERT_UNK_ID = 100
BERT_CLS_ID = 101
BERT_SEP_ID = 102
BERT_MASK_ID = 103


def convert_subword_to_word(sentence):
    return sentence.replace(' ##', '')


class Vocabulary(object):
    def __init__(self,
                 vocab_fname: str = None,
                 vocab_dict: Dict[str, int] = None,
                 num_oov_buckets: int = 1,
                 unk_token: str = _PARLAI_UNK):
        if vocab_fname is None and vocab_dict is None:
            raise ValueError("One of 'vocab_fname' or 'vocab_dict' should not be None")
        if vocab_fname and vocab_dict:
            raise ValueError("Only one of 'vocab_fname' or 'vocab_dict' can have value")

        if vocab_fname:
            raise NotImplementedError
        elif vocab_dict:
            strings = []
            indices = []
            for key, value in vocab_dict.items():
                strings.append(key)
                indices.append(value)
            self.string_to_index_table = tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=strings, values=indices, key_dtype=tf.string, value_dtype=tf.int64
                ), num_oov_buckets, lookup_key_dtype=tf.string
            )
            self.index_to_string_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=indices, values=strings, key_dtype=tf.int64, value_dtype=tf.string
                ), default_value=unk_token
            )

        self._num_oov_buckets = num_oov_buckets
        self._unk_token = unk_token

    def string_to_index(self, keys):
        return self.string_to_index_table.lookup(keys)

    def index_to_string(self, keys):
        if keys.dtype == tf.int32:
            keys = tf.cast(keys, tf.int64)
        return self.index_to_string_table.lookup(keys)
