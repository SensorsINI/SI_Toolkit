import tensorflow as tf
import tensorflow_probability as tfp


def interpolate_tf(y_ref, step=10, axis=1):
    range_stop = (y_ref.shape[axis]-1)*step + 1
    t_interp = tf.cast(tf.range(range_stop), tf.float32)
    interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], y_ref, axis=1)
    return interp
