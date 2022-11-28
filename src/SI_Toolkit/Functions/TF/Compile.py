import logging
import platform

import tensorflow as tf
import torch

from Control_Toolkit.others.globals_and_utils import get_logger
log=get_logger(__name__)

from SI_Toolkit.computation_library import ComputationLibrary



try:
    from SI_Toolkit_ASF import GLOBALLY_DISABLE_COMPILATION, USE_JIT_COMPILATION
except ImportError:
    log.warn("No compilation option set in SI_Toolkit_ASF/__init.py__. Setting GLOBALLY_DISABLE_COMPILATION to True.")
    GLOBALLY_DISABLE_COMPILATION = True

def tf_function_jit(func):
    return tf.function(func=func, jit_compile=True,)


def tf_function_experimental(func):
    return tf.function(func=func, experimental_compile=True)


def identity(func):
    return func


if GLOBALLY_DISABLE_COMPILATION:
    log.info('TensorFlow compilation is disabled by GLOBALLY_DISABLE_COMPILATION=True')
    CompileTF = identity
else:
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':  # For M1 Apple processor
        CompileTF = tf.function
    elif not USE_JIT_COMPILATION:
        CompileTF = tf.function
    else:
        CompileTF = tf_function_jit
    log.info(f'using {CompileTF} compilation')
        # CompileTF = tf_function_experimental # Should be same as tf_function_jit, not appropriate for newer version of TF

def CompileAdaptive(fun):
    """ TODO add docstring to explain what it does and where it is used
    """
    instance = fun.__self__
    assert hasattr(instance, "lib"), "Instance with this method has no computation library defined"
    computation_library: "type[ComputationLibrary]" = instance.lib
    lib_name = computation_library.lib

    if GLOBALLY_DISABLE_COMPILATION:
        return identity(fun)
    elif lib_name == 'TF':
        return CompileTF(fun)
    else:
        log.warning(f'JIT compilation for {lib_name} not yet implemented.')
        return identity(fun)
