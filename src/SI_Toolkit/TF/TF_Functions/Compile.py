import tensorflow as tf
import platform
from SI_Toolkit_ASF_global import GLOBALLY_DISABLE_COMPILATION

def tf_function_jit(func):
    return tf.function(func=func, jit_compile=True)


def tf_function_experimental(func):
    return tf.function(func=func, experimental_compile=True)


def identity(func):
    return func


if GLOBALLY_DISABLE_COMPILATION:
    Compile = identity
else:
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':  # For M1 Apple processor
        Compile = tf.function
    else:
        Compile = tf_function_jit
        # Compile = tf_function_experimental # Should be same as tf_function_jit, not appropriate for newer version of TF
