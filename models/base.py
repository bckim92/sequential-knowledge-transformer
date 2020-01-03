from typing import Dict, Union

import tensorflow as tf
import colorlog
import numpy as np
from tensorflow.python.framework.ops import EagerTensor  # pylint: disable=no-name-in-module
from tensorflow.python.framework.ops import Tensor  # pylint: disable=no-name-in-module

from data.vocabulary import Vocabulary


class BaseModel(tf.keras.Model):
    """
    This abstract class represents a model to be trained. Rather than relying
    completely on the Tensorflow's Keras Module, we modify the output sepc of
    ``call`` to be a dictionary.

    Even though, it is still compatible with other keras model.

    To use this class, we must implement following methods.
    - def __init__()
        Define your layers
    - def call(self, inputs)
        Implement the model's forward pass
    - (optionally) def compute_output_shape
    """
    def __init__(self,
                 hparams,
                 vocabulary: Vocabulary,
                 name: str) -> None:
        super().__init__(name=name)
        self.hparams = hparams
        self.vocabulary = vocabulary

    def call(self, inputs, training: bool = True) -> Dict[str, Union[Tensor, EagerTensor]]:  # pylint: disable=arguments-differ
        raise NotImplementedError

    def print_model(self):
        for var in self.trainable_variables:
            print(f"  {var.name}, {var.shape}, {var.device}")

    def pad_word_outputs(self, outputs, output_max_length):
        batch_size = tf.shape(outputs)[0]
        padding = tf.zeros([batch_size, self.hparams.max_length - output_max_length + 1], dtype=tf.int64)
        padded_words = self.vocabulary.index_to_string(tf.concat([tf.cast(outputs, tf.int64), padding], axis=1))
        return padded_words

    @property
    def embedding(self) -> tf.keras.layers.Layer:
        return self._embedding  # pylint: disable=no-member
