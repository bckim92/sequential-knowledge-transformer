import tensorflow as tf


def single_rnn_cell(units: int,
                    cell_type: str = "gru",
                    name: str = None):
    if cell_type == "gru":
        # Masking is not supported for CuDNN RNNs
        return tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            name=name)
    elif cell_type == "cudnn_gru":
        return tf.compat.v1.keras.layers.CuDNNGRU(
            units,
            return_sequences=True,
            return_state=True,
            name=name)
    elif cell_type == "gru_cell":
        #return tf.keras.layers.GRUCell(
        #    units,
        #    name=name
        #)
        # Use this for decoder
        return tf.nn.rnn_cell.GRUCell(
            units,
            name=name
        )
    else:
        raise ValueError


class RnnEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 units: int,
                 cell_type: str = "gru",
                 name: str = None):
        super().__init__(name=name)
        self._units = units
        self._cell_type = cell_type
        self._name = name

    def build(self, input_shape):
        rnn_cell = single_rnn_cell(self._units, self._cell_type)
        self.birnn_cell = tf.keras.layers.Bidirectional(rnn_cell)
        super().build(input_shape)

    def call(self, x, initial_state=None):
        outputs, fw_state, bw_state = self.birnn_cell(x, initial_state=initial_state)
        return outputs, fw_state, bw_state

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        batch_size = shape[0]
        shape[-1] = self._units * 2
        return [tf.TensorShape(shape),
                tf.TensorShape([batch_size, self._units]),
                tf.TensorShape([batch_size, self._units])]

    def get_config(self):
        return {
            "units": self._units,
            "cell_type": self._cell_type
        }

    def compute_mask(self, inputs, mask):
        return self.birnn_cell.compute_mask(inputs, mask)
