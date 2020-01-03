import tensorflow as tf
from tensorflow.python.keras.layers import convolutional


class WeightNormDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            'scale',
            shape=[self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        original_kernel = self.kernel
        self.kernel = self.normalized_kernel
        outputs = super().call(inputs)
        self.kernel = original_kernel
        return outputs

    @property
    def normalized_kernel(self):
        return tf.expand_dims(self.scale, axis=0) * \
            tf.nn.l2_normalize(self.kernel, axis=[0])


class WeightNormConv(convolutional.Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(self.filters,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        super().build(input_shape)

    def call(self, inputs):
        original_kernel = self.kernel
        self.kernel = self.normalized_kernel
        outputs = super().call(inputs)
        self.kernel = original_kernel
        return outputs

    @property
    def normalized_kernel(self):
        return tf.reshape(self.scale, [1, 1, -1]) * \
            tf.nn.l2_normalize(self.kernel, axis=[0, 1])


class WeightNormConv1D(WeightNormConv):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(rank=1, filters=filters,
                         kernel_size=kernel_size, **kwargs)
