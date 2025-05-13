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

import torch, functools, inspect
def _torch_compile(fn, *, top_only=True):
    """
    Wrap *fn* for torch.compile while printing a single line when
    the *top* compiled function is entered / left.

    Set top_only=False if you really want every nested call.
    """
    qual = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if top_only and inspect.currentframe().f_back.f_code.co_name != wrapper.__name__:
            # Nested call inside the same compiled graph: skip
            return fn(*a, **kw)
        print(f"[compile ▶] {qual}")
        out = fn(*a, **kw)
        print(f"[compile ▲] {qual}")
        return out

    return torch.compile(wrapper, fullgraph=False, dynamic=False)



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
