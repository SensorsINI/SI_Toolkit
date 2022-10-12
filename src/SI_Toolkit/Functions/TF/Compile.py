import logging
import tensorflow as tf
import platform

# from Control_Toolkit.others.environment import ComputationLibrary # Fixme: throws an error of circular import

try:
    from SI_Toolkit_ASF import GLOBALLY_DISABLE_COMPILATION, USE_JIT_COMPILATION
except ImportError:
    logging.warn("No compilation option set in SI_Toolkit_ASF. Setting GLOBALLY_DISABLE_COMPILATION to True.")
    GLOBALLY_DISABLE_COMPILATION = True

def tf_function_jit(func):
    return tf.function(func=func, jit_compile=True)


def tf_function_experimental(func):
    return tf.function(func=func, experimental_compile=True)


def identity(func):
    return func


if GLOBALLY_DISABLE_COMPILATION:
    CompileTF = identity
else:
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':  # For M1 Apple processor
        CompileTF = tf.function
    elif not USE_JIT_COMPILATION:
        CompileTF = tf.function
    else:
        CompileTF = tf_function_jit
        # CompileTF = tf_function_experimental # Should be same as tf_function_jit, not appropriate for newer version of TF

def CompileAdaptive(fun):
    instance = fun.__self__
    assert hasattr(instance, "lib"), "Instance with this method has no computation library defined"
    # computation_library: ComputationLibrary = instance.lib # Fixme: throws an error of circular import
    computation_library = instance.lib
    lib_name = computation_library.lib

    if GLOBALLY_DISABLE_COMPILATION:
        return identity(fun)
    elif lib_name == 'TF':
        return CompileTF(fun)
    else:
        raise NotImplementedError("...")
