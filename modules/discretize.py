import tensorflow as tf
import tensorflow_probability as tfp


def gumbel_softmax(temperature, probs=None, hard=True):
    num_classes = tf.shape(probs)[-1]

    sampler = tfp.distributions.RelaxedOneHotCategorical(
        temperature, probs=probs)
    sample = sampler.sample()
    if hard:
        sample_hard = tf.one_hot(tf.argmax(sample, axis=-1),
                                 num_classes, dtype=tf.float32)
        sample_onehot = tf.stop_gradient(sample_hard - sample) + sample
        sample_idx = tf.cast(tf.argmax(sample_onehot, axis=-1), dtype=tf.int32)
        return sample_onehot, sample_idx
    else:
        return sample
