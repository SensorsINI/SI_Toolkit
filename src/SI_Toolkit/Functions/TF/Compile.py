import platform
import os
import importlib
import logging

from SI_Toolkit.computation_library import ComputationLibrary, ComputationClasses

from SI_Toolkit.load_and_normalize import load_yaml
config_compilation = load_yaml(os.path.join('SI_Toolkit_ASF', 'CONFIG_COMPILATION.yml'))
GLOBALLY_DISABLE_COMPILATION = config_compilation['GLOBALLY_DISABLE_COMPILATION']
USE_JIT_COMPILATION = config_compilation['USE_JIT_COMPILATION']

def tf_function_jit(func):
    import tensorflow as tf
    return tf.function(func=func, jit_compile=True)


def tf_function_experimental(func):
    import tensorflow as tf
    return tf.function(func=func, experimental_compile=True)


def identity(func):
    return func


if GLOBALLY_DISABLE_COMPILATION:
    CompileTF = identity
else:
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':  # For M1 Apple processor
        import tensorflow as tf
        CompileTF = tf.function
    elif not USE_JIT_COMPILATION:
        import tensorflow as tf
        CompileTF = tf.function
    else:
        CompileTF = tf_function_jit
        # CompileTF = tf_function_experimental # Should be same as tf_function_jit, not appropriate for newer version of TF


def _torch_compile(fn):
    import torch
    from torch._inductor import config
    import os  # for retrieving the number of CPU cores available

    # Set compile threads to the maximum available CPU cores (fallback to 1 if detection fails)
    config.compile_threads = os.cpu_count()//2 or 1

    compiled_fn = torch.compile(
        fn,
        fullgraph=False,
        dynamic=False,
        backend="inductor"
    )

    return compiled_fn




def CompileAdaptive(arg=None):
    if isinstance(arg, ComputationClasses):

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
        return _torch_compile(fun)
    else:
        return identity(fun)
