import tensorflow as tf
import colorlog

from models import BaseModel
from models.transformer import model_utils
from models.transformer import attention_layer
from models.transformer.attention_layer import _float32_softmax
from models.transformer import ffn_layer
from models.transformer import beam_search
from data import vocabulary as data_vocab
from utils.config_utils import add_argument

class TransformerEncoder(BaseModel):
    """Transformer Encoder model with Keras.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continuous
    representation, and the decoder uses the encoder output ot generate
    probabilities for the output sequence.
    """

    def __init__(self,
                 hparams,
                 vocabulary,
                 input_embedding_layer,
                 output_embedding_layer):
        """Initialize layers to build Transformer encoder model.

        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            name: name of the model.
        """
        super().__init__(hparams, vocabulary, "TransformerEncoder")
        self._input_embedding = input_embedding_layer
        self._output_embedding = output_embedding_layer
        self.encoder_stack = EncoderStack(hparams)

    def get_config(self):
        return {
            "hparams": self.hparams,
        }

    def call(self, inputs, inputs_embedding, training: bool = True):
        """Calculate target logits or inferred target sequences.

        Args:
            inputs: int tensor with shape [batch_size, input_length].
            training: boolean, whether in training mode or not.

        Returns:
            encoder outputs: float tensor with shape
            [batch_size, input_length, word_embed_size]
        Even when float16 is used, the output tensor(s) are always float32.
        """
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("TransformerEncoder"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = model_utils.get_padding_bias(inputs)

            # Run the inputs through the encoder layer to map the symbol
            # representations to coninuous representations.
            encoder_outputs = self.encode(inputs, inputs_embedding, attention_bias, training)

            return encoder_outputs


    def encode(self, inputs, inputs_embedding, attention_bias, training):
        """Generate continuous representation for inputs.

        Args:
            inputs: int tensor with shape [batch_size, input_length].
            attention_bias: float tensort with shape [batch_size, 1, 1, input_length].
            training: boolean, whether in training mode or not.

        Returns:
            float tensor with shape [batch_size, input_length, word_embed_size]
        """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # appling dropout.
            #embedded_inputs = self._input_embedding(inputs)
            embedded_inputs = inputs_embedding
            embedded_inputs = tf.cast(embedded_inputs, tf.float32)
            inputs_padding = model_utils.get_padding(inputs)
            attention_bias = tf.cast(attention_bias, tf.float32)

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.hparams.word_embed_size)
                pos_encoding = tf.cast(pos_encoding, tf.float32)
                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs,
                    noise_shape=[tf.shape(encoder_inputs)[0], 1, tf.shape(encoder_inputs)[2]],
                    rate=self.hparams.layer_postprocess_dropout)
            return self.encoder_stack(
                encoder_inputs, attention_bias, inputs_padding, training=training)

class TransformerDecoder(BaseModel):
    """Transformer Decoder model with Keras.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continuous
    representation, and the decoder uses the encoder output ot generate
    probabilities for the output sequence.
    """

    def __init__(self,
                 hparams,
                 vocabulary,
                 input_embedding_layer,
                 output_embedding_layer):
        """Initialize layers to build Transformer Decoder model.

        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            name: name of the model.
        """
        super().__init__(hparams, vocabulary, "TransformerDecoder")
        self._input_embedding = input_embedding_layer
        self._output_embedding =  output_embedding_layer
        self.decoder_stack = DecoderStack(hparams)
        if hparams.use_copy_decoder:
            self._copy_q_layer = tf.keras.layers.Dense(
                hparams.word_embed_size, use_bias=False, name="copy_q")
            self._copy_k_layer = tf.keras.layers.Dense(
                hparams.word_embed_size, use_bias=False, name="copy_k")
            self._copy_v_layer = tf.keras.layers.Dense(
                hparams.word_embed_size, use_bias=False, name="copy_v")
            self._copy_layer = tf.keras.layers.Dense(
                1, activation=tf.math.sigmoid, name="copy_layer", use_bias=False)
            # self._copy_q_layer = tf.keras.layers.Dense(
            #     hparams.word_embed_size, use_bias=False, name="copy_q", kernel_initializer='glorot_normal')
            # self._copy_k_layer = tf.keras.layers.Dense(
            #     hparams.word_embed_size, use_bias=False, name="copy_k", kernel_initializer='glorot_normal')
            # self._copy_v_layer = tf.keras.layers.Dense(
            #     hparams.word_embed_size, use_bias=False, name="copy_v", kernel_initializer='glorot_normal')
            # self._copy_layer = tf.keras.layers.Dense(
            #     1, activation=tf.math.sigmoid, name="copy_layer", use_bias=False, kernel_initializer='glorot_normal')

    def get_config(self):
        return {
            "hparams": self.hparams,
        }

    def call(self, mixed_inputs, encoder_outputs, targets = None,
             targets_embedding = None, training: bool = True):
        """Calculate target logits or inferred target sequences.
        Args:
            First item, mixed inputs: int tensor with shape
            [batch_size, knowledge_max_length + context_length]
            Second item, encoder_outputs: float tensor with shape
            [batch_size, sentence_max_length, word_embed_size].
            Third item (optional), targets: int tensor with shape
            [batch_size, target_length].
            training: boolean, whether in training mode or not.

        Returns:
            If targets is defined, then return logits for each word in the target
            sequence. float tensor with shape [batch_size, target_length, vocab_size]
            If target is none, then generate output sequence one token at a time.
            returns a dictionary {
                outputs: [batch_size, decoded length]
                scores: [batch_size, float]}
        Even when float16 is used, the output tensor(s) are always float32.
        """
        with tf.name_scope("TransformerDecoder"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = model_utils.get_padding_bias(mixed_inputs)
            if not training:
                return self.predict(mixed_inputs, encoder_outputs, attention_bias, training)
            else:
                logits, sample_ids = self.decode(mixed_inputs, targets, targets_embedding,
                                                 encoder_outputs, attention_bias,
                                                 training)
                return logits, sample_ids

    def decode(self, mixed_inputs, targets, targets_embedding, encoder_outputs, attention_bias, training):
        """Generate logits for each value in the target sequence.

        Args:
            mixed_inputs: input values of context and chosen knowledge. Int tensor with shape
            [batch_size, mixed_input_length]
            targets: target values for the output sequence. Int tensor with shape
            [batch_size, target_length]
            encoder_outputs: continuous representation of input sequence. float tensor
            with shape [batch_size, sentence_max_length, word_embed_size]
            attention_bias: float tensor with shape [batch_size, 1, 1, sentence_max_length]
            training: boolean, whether in training mode or not.

        Returns:
            float32 tensor with shape [batch_size, target_length, vocab_size]
        """
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            #decoder_inputs = self._input_embedding(targets)
            decoder_inputs = targets_embedding
            decoder_inputs = tf.cast(decoder_inputs, tf.float32)
            attention_bias = tf.cast(attention_bias, tf.float32)
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = decoder_inputs[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.hparams.word_embed_size)
                pos_encoding = tf.cast(pos_encoding, tf.float32)
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs,
                    noise_shape=[tf.shape(decoder_inputs)[0], 1, tf.shape(decoder_inputs)[2]],
                    rate=self.hparams.layer_postprocess_dropout)

            # Run values
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length, dtype=tf.float32)
            decoder_outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias,
                training=training)

            if self.hparams.use_copy_decoder:
                logits = self.copy_decode(mixed_inputs, encoder_outputs,
                                          decoder_outputs, attention_bias, training)
            else:
                logits = self._output_embedding(decoder_outputs, mode="linear")
                logits = tf.cast(logits, tf.float32)
            sample_ids = tf.argmax(logits, axis=2)

            return logits, sample_ids

    def copy_decode(self, mixed_inputs, encoder_outputs, decoder_outputs, attention_bias, training):
        """ Generate softmax values of logits in the target sequence.

        Args: Same as decode function's arguments
            - mixed_inputs: input values of context and chosen knowledge. Int tensor with shape
            [batch_size, mixed_input_length]
            - encoder_outputs: continuous representation of input sequence. float tensor
            with shape [batch_size, sentence_max_length, word_embed_size]
            - decoder_outputs: continuous representaiton of output sequence. float tensor
            with shape [batch_size, target_length - 1, word_embed_size]
            - attention_bias: float tensor with shape [batch_size, 1, 1, sentence_max_length]
            training: boolean, whether in training mode or not.
        Returns:
            float32 tensor with shape [batch_size, target_length, vocab_size]
        """
        with tf.name_scope("copy_decode"):
            colorlog.warning("Use pointer-generator mechanism. \
                             Note that output is not logit but softmax.")
            if training:
                batch_size = tf.shape(mixed_inputs)[0]
                # batch_size = self.hparams.batch_size
            else:
                batch_size = tf.shape(mixed_inputs)[0] * self.hparams.beam_size
                # batch_size = self.hparams.batch_size * self.hparams.beam_size

            w_q = self._copy_q_layer
            w_k = self._copy_k_layer
            w_v = self._copy_v_layer

            q = w_q(decoder_outputs)
            k = w_k(encoder_outputs)
            v = w_v(encoder_outputs)

            # Codes for multi heads attention, but not necessary.

            q = self.decoder_stack.layers[-1][1].layer.split_heads(q)
            k = self.decoder_stack.layers[-1][1].layer.split_heads(k)
            v = self.decoder_stack.layers[-1][1].layer.split_heads(v)

            depth = (self.hparams.word_embed_size // self.hparams.num_heads)
            q *= depth ** -0.5

            a_t = tf.matmul(q, k, transpose_b=True)
            a_t += attention_bias
            # [batch_size, num_heads, target_length - 1, mixed_input_length]
            p_att = _float32_softmax(a_t, name="p_copy")
            if training:
                p_att = tf.nn.dropout(p_att,
                                      noise_shape=[tf.shape(p_att)[0], tf.shape(p_att)[1], 1, 1],
                                      rate=self.hparams.attention_dropout)

            # [batch_size, num_heads, target_length - 1, depth]
            hidden = tf.matmul(p_att, v)
            # [batch_size, target_length, word_embed_size]
            p_att = p_att[:,0]
            hidden = self.decoder_stack.layers[-1][1].layer.combine_heads(hidden)
            hidden = self.decoder_stack.layers[-1][1].layer.output_dense_layer(hidden)
            # feed forward network
            hidden = self.decoder_stack.layers[-1][2](hidden, training=training)
            hidden = self.decoder_stack.output_normalization(hidden)
            # [batch_size, target_length - 1, vocab_size]
            p_vocab = _float32_softmax(self._output_embedding(decoder_outputs, mode="linear"))
            # p_vocab = _float32_softmax(self._embedding(decoder_outputs, mode="linear"))

            # matching (p_att.shape) to (p_vocab.shape)
            initial_indices = tf.tile(mixed_inputs[:, tf.newaxis, :], [1, tf.shape(p_vocab)[1], 1])
            i1, i2 = tf.meshgrid(tf.range(batch_size),
                     tf.range(tf.shape(p_vocab)[1]), indexing="ij")
            i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, tf.shape(p_att)[2]])
            i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, tf.shape(p_att)[2]])
            # [batch_size, target_length - 1, mixed_input_length, 3]
            indices = tf.stack([i1, i2, initial_indices], axis=-1)
            # [batch_size, target_length - 1, vocab_size]
            p_att = tf.scatter_nd(indices, p_att, shape=tf.shape(p_vocab))

            p_gen = self._copy_layer(hidden)
            # [batch_size, target_length - 1, vocab_size]
            p_gen = tf.tile(p_gen, [1, 1, self.hparams.vocab_size])
            # [batch_size, target_length - 1, vocab_size]
            p_word = (1 - p_gen) * p_vocab + p_gen * p_att

            return p_word

    def predict(self, mixed_inputs, encoder_outputs, encoder_decoder_attention_bias, training):
        """Return predicted sequence."""
        # Currently, we always do prediction in float32
        # TODO(reedwm): Add float16 support.
        encoder_outputs = tf.cast(encoder_outputs, tf.float32)
        batch_size = tf.shape(encoder_outputs)[0]
        max_decode_length = self.hparams.max_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length, training)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32) + data_vocab.BERT_CLS_ID
        eos_id = data_vocab.BERT_SEP_ID

        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.hparams.word_embed_size]),
                "v": tf.zeros([batch_size, 0, self.hparams.word_embed_size])
            } for layer in range(self.hparams.num_layers)
        }

        # Add mixed input, encoder output and attention bias to the cache.
        cache["mixed_inputs"] = mixed_inputs
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.hparams.vocab_size,
            beam_size=self.hparams.beam_size,
            alpha=self.hparams.beam_search_alpha,
            max_decode_length=max_decode_length,
            eos_id=eos_id,
            use_copy_decoder=self.hparams.use_copy_decoder)

        # Get the top sequence for each batch element
        # [batch_size, max_decode_length]
        top_decoded_ids = decoded_ids[:, 0, 1:]

        # [batch_size]
        top_scores = scores[:, 0]

        top_decoded_max_length = tf.shape(top_decoded_ids)[1]

        return top_decoded_ids, top_scores, top_decoded_max_length

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = model_utils.get_position_encoding(
            max_decode_length + 1, self.hparams.word_embed_size)
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

            Args:
              ids: Current decoded sequences. int tensor with shape [batch_size *
                beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self._input_embedding(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input,
                cache.get("encoder_outputs"),
                self_attention_bias,
                cache.get("encoder_decoder_attention_bias"),
                training=training,
                cache=cache)

            if self.hparams.use_copy_decoder:
                # [batch_size, 1, vocab_size]
                logits = self.copy_decode(cache.get("mixed_inputs"),
                                          cache.get("encoder_outputs"),
                                          decoder_outputs,
                                          cache.get("encoder_decoder_attention_bias"),
                                          training=training)
            else:
                logits = self._output_embedding(decoder_outputs, mode="linear")

            # [batch_size, vocab_size]
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

class LayerNormalization(tf.keras.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, word_embed_size):
        super(LayerNormalization, self).__init__()
        self.word_embed_size = word_embed_size

    def build(self, input_shape):
        """Builds the layer"""
        # Passing experimental_autocast=False casues these variables to not be
        # automatically casted to fp16 when mixed precision is used. Since we use
        # float32 in call() for numeric stability, we do not want variables to be
        # casted to fp16.
        self.scale = self.add_weight(
            "layer_norm_scale",
            shape=[self.word_embed_size],
            dtype="float32",
            initializer=tf.ones_initializer(),
            experimental_autocast=False)
        self.bias = self.add_weight(
            "layer_norm_bias",
            shape=[self.word_embed_size],
            dtype="float32",
            initializer=tf.zeros_initializer(),
            experimental_autocast=False)
        super(LayerNormalization, self).build(input_shape)

    def get_config(self):
        return {
            "word_embed_size": self.word_embed_size,
        }

    def call(self, x, epsilon=1e-6):
        input_dtype = x.dtype
        if input_dtype == tf.float16:
            x = tf.cast(x, tf.float32)
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return tf.cast(norm_x * self.scale + self.bias, input_dtype)

class PrePostProcessingWrapper(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, hparams):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.hparams = hparams
        self.postprocess_dropout = hparams.layer_postprocess_dropout

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = LayerNormalization(self.hparams.word_embed_size)
        super(PrePostProcessingWrapper, self).build(input_shape)

    def get_config(self):
        return {
            "hparams": self.hparams,
        }

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]

        y = self.layer_norm(x)
        # Get layer output
        y = self.layer(y, *args, **kwargs)

        if training:
            y = tf.nn.dropout(y,
                              noise_shape=[tf.shape(y)[0], 1, tf.shape(y)[2]],
                              rate=self.postprocess_dropout)
        return x + y

class EncoderStack(tf.keras.layers.Layer):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
        1. Self-attention layer
        2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, hparams):
        super(EncoderStack, self).__init__()
        self.hparams = hparams
        self.layers = []

    def build(self, input_shape):
        """Builds the encoder stack."""
        hparams = self.hparams
        for _ in range(hparams.num_layers):
            # Create sublayers for each laer.
            self_attention_layer = attention_layer.SelfAttention(
                hparams.word_embed_size, hparams.num_heads,
                hparams.attention_dropout)
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                hparams.word_embed_size, hparams.filter_size,
                hparams.relu_dropout)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, hparams),
                PrePostProcessingWrapper(feed_forward_network, hparams)
            ])

        # Create final layer normalization laeyr.
        self.output_normalization = LayerNormalization(hparams.word_embed_size)
        super(EncoderStack, self).build(input_shape)

    def get_config(self):
        return {
            "hparams": self.hparams,
        }

    def call(self, encoder_inputs, attention_bias, inputs_padding, training):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
            1, input_length]
          inputs_padding: tensor with shape [batch_size, input_length], inputs with
            zero paddings.
          training: boolean, whether in training mode or not.

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.name_scope("layer_%d" %n):
                with tf.name_scope("self_attention"):
                    encoder_inputs = self_attention_layer(
                        encoder_inputs, attention_bias, training=training)
                with tf.name_scope("ffn"):
                    encoder_inputs = feed_forward_network(
                        encoder_inputs, training=training)

        return self.output_normalization(encoder_inputs)

class DecoderStack(tf.keras.layers.Layer):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
        1. Self-attention layer
        2. Multi-headed attention layer combining encoder outputs with results from
           the previous self-attention layer.
        3. Feedforward network (2 fully-conneced layers)
    """

    def __init__(self, hparams):
        super(DecoderStack, self).__init__()
        self.hparams = hparams
        self.layers = []

    def build(self, input_shape):
        """Builds the decoder stack."""
        hparams = self.hparams
        for _ in range(hparams.num_layers):
            self_attention_layer = attention_layer.SelfAttention(
                hparams.word_embed_size, hparams.num_heads,
                hparams.attention_dropout)
            enc_dec_attention_layer = attention_layer.Attention(
                hparams.word_embed_size, hparams.num_heads,
                hparams.attention_dropout)
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                hparams.word_embed_size, hparams.filter_size,
                hparams.relu_dropout)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, hparams),
                PrePostProcessingWrapper(enc_dec_attention_layer, hparams),
                PrePostProcessingWrapper(feed_forward_network, hparams)
            ])
        self.output_normalization = LayerNormalization(hparams.word_embed_size)
        super(DecoderStack, self).build(input_shape)

    def get_config(self):
        return {
            "hparams": self.hparams,
        }

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, training, cache=None):
        """Return the output of the decoder layer stacks.

        Args:
            decoder_inputs: tensor with shape [batch_size, target_length, word_embed_size]
            encoder_outputs: tensor with shape [batch_size, sentence_max_length, word_embed_size]
            decoder_self_attention_bias: bias for decoder self-attention layer. [1, 1,
                target_len, target_length]
            attention_bias: bias for encoder-decoder attention layer. [batch_size, 1,
                1, sentence_max_length]
            training: boolean, whether in training mode or not.
            cache: (Used for fast decoding) A nested dictionary storing previous
                decoder self-attention values. The items are:
                    {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                                "v": tensor with shape [batch_size, i, value_channels]},
                                ...}
        Returns:
            output of decoder layer stack.
            float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.name_scope(layer_name):
                with tf.name_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs,
                        decoder_self_attention_bias,
                        training=training,
                        cache=layer_cache)
                with tf.name_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs,
                        encoder_outputs,
                        attention_bias,
                        training=training)
                with tf.name_scope("ffn"):
                    decoder_inputs = feed_forward_network(
                        decoder_inputs, training=training)

        return self.output_normalization(decoder_inputs)
