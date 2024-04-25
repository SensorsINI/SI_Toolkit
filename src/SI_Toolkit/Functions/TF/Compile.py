import logging
import platform
import os

import tensorflow as tf
import torch

from SI_Toolkit.computation_library import ComputationLibrary

from SI_Toolkit.load_and_normalize import load_yaml
config_compilation = load_yaml(os.path.join('SI_Toolkit_ASF', 'CONFIG_COMPILATION.yml'))
GLOBALLY_DISABLE_COMPILATION = config_compilation['GLOBALLY_DISABLE_COMPILATION']
USE_JIT_COMPILATION = config_compilation['USE_JIT_COMPILATION']

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


def CompileAdaptive(arg=None):
    if isinstance(arg, type) and issubclass(arg, ComputationLibrary):

        computation_library = arg

        def decorator_factory(fun):
            return decorator(fun, computation_library)

        return decorator_factory

    elif callable(arg):  # No arguments provided, the only argument is the function to be decorated
        function = arg
        return decorator(function)

    else:
        # Handle other cases or raise an error.
        raise TypeError("Invalid argument to CompileAdaptive")



def decorator(fun, computation_library=None):
    if computation_library is None:
        if hasattr(fun, "__self__"):
            instance = fun.__self__
            assert hasattr(instance, "lib"), "Instance with this method has no computation library defined"
            computation_library: "type[ComputationLibrary]" = instance.lib

    if computation_library is not None:
        lib_name = computation_library.lib
    else:
        lib_name = None
        print("Not compiling")

    if GLOBALLY_DISABLE_COMPILATION:
        return identity(fun)
    elif lib_name == 'TF':
        return CompileTF(fun)
    elif lib_name == 'Pytorch':
        print('Jit compilation for Pytorch not yet implemented.')
        return identity(fun)
    else:
        return identity(fun)
