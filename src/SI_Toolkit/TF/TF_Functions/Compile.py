import tensorflow as tf


def tf_function_jit(func):
    return tf.function(func=func, jit_compile=True)


def tf_function_experimental(func):
    return tf.function(func=func, experimental_compile=True)


def identity(func):
    return func

Compile = tf.function
# Compile = tf_function_jit
# Compile = tf_function_experimental
# Compile = identity
