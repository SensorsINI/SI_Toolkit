from typing import Callable, Optional, Union, Sequence, Any

import functools

from math import pi


# TensorType = Union[np.ndarray, tf.Tensor, torch.Tensor]
TensorType = Union["np.ndarray", "tf.Tensor", "torch.Tensor"]
RandomGeneratorType = Union["np.random.Generator", "tf.random.Generator", "torch.Generator"]
NumericType = Union[float, int]


def select_library(lib_name):
    if lib_name == 'Numpy':
        return NumpyLibrary()
    elif lib_name == 'TF':
        return TensorFlowLibrary()
    elif lib_name == 'Pytorch':
        return PyTorchLibrary()
    else:
        raise ValueError(f"{lib_name} is not a valid library name.")


def set_device_general(device_name: str, library: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    A decorator to set the computation device for the decorated function.

    Parameters:
    - device_name: str - The name of the device (e.g., '/cpu:0', '/gpu:0').

    Returns:
    - A callable decorator that wraps the original function, ensuring it runs on the specified device.
    """
    library = library.lower()  # Normalize the library name for case-insensitive comparison

    if library == 'numpy':

        import numpy as np

        # Numpy does not require device management, so this is a no-op
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper

    elif library == 'tf':

        import tensorflow as tf

        devices = tf.config.list_physical_devices()
        if device_name not in [d.name for d in devices] and '/physical_'+device_name[1:] not in [d.name for d in devices]:
            raise ValueError(f"Requested device {device_name} not found in the list of physical devices: {devices}")

        # TensorFlow device management
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with tf.device(device_name):
                    return func(*args, **kwargs)
            return wrapper

    elif library == 'pytorch':

        import torch

        devices = [torch.device(f'cuda:{i}') for i in
                   range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device('cpu')]
        target_device = torch.device(device_name)  # Create a torch.device object for comparison

        if target_device not in devices:
            raise ValueError(f"Requested device {device_name} not found in the list of physical devices: {devices}")

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Process the first argument separately if it's a class instance with a .to() method
                if args and hasattr(args[0], 'to'):
                    first_arg = args[0].to(target_device) if getattr(args[0], 'device', None) != target_device else args[0]
                    args = (first_arg,) + args[1:]

                # Then, handle all the args and kwargs
                new_args = [
                    arg.to(target_device) if isinstance(arg, torch.Tensor) and arg.device != target_device else arg
                    for arg in args
                ]
                new_kwargs = {
                    k: (v.to(target_device) if isinstance(v, torch.Tensor) and v.device != target_device else v) for
                    k, v in kwargs.items()}
                return func(*new_args, **new_kwargs)

            return wrapper
    else:
        raise ValueError('Invalid library name')

    return decorator


class ComputationLibrary:
    lib = None
    set_device: Callable[[str], Callable[[Callable[..., Any]], Callable[..., Any]]] = None
    reshape: Callable[[TensorType, "tuple[int, ...]"], TensorType] = None
    permute: Callable[[TensorType, "tuple[int]"], TensorType] = None
    newaxis = None
    shape: Callable[[TensorType], "list[int]"] = None
    to_numpy: Callable[[TensorType], "numpy.ndarray"] = None
    to_variable: Callable[[TensorType, type], "numpy.ndarray"] = None
    to_tensor: Callable[[TensorType, type], TensorType] = None
    constant: Callable[[Union[float, TensorType], type], Union[float, TensorType]] = None
    unstack: Callable[[TensorType, int, int], "list[TensorType]"] = None
    ndim: Callable[[TensorType], int] = None
    clip: Callable[[TensorType, float, float], TensorType] = None
    sin: Callable[[TensorType], TensorType] = None
    asin: Callable[[TensorType], TensorType] = None
    cos: Callable[[TensorType], TensorType] = None
    tan: Callable[[TensorType], TensorType] = None
    tanh: Callable[[TensorType], TensorType] = None
    exp: Callable[[TensorType], TensorType] = None
    reciprocal: Callable[[TensorType], TensorType] = None
    squeeze: Callable[[TensorType], TensorType] = None
    unsqueeze: Callable[[TensorType, int], TensorType] = None
    stack: Callable[["list[TensorType]", int], TensorType] = None
    cast: Callable[[TensorType, type], TensorType] = None
    floormod: Callable[[TensorType], TensorType] = None
    floor: Callable[[TensorType], TensorType] = None
    ceil: Callable[[TensorType], TensorType] = None
    rint: Callable[[TensorType], TensorType] = None
    float32 = None
    float64 = None
    int32 = None
    int64 = None
    bool = None
    tile: Callable[[TensorType, Sequence[int]], TensorType] = None
    repeat: Callable[[TensorType, int, int], TensorType] = None
    gather: Callable[[TensorType, TensorType, int], TensorType] = None
    gather_last: Callable[[TensorType, TensorType], TensorType] = None
    arange: Callable[[Optional[NumericType], NumericType, Optional[NumericType]], TensorType] = None
    zeros: Callable[["tuple[int]"], TensorType] = None
    zeros_like: Callable[[TensorType], TensorType] = None
    ones: Callable[["tuple[int]"], TensorType] = None
    ones_like: Callable[[TensorType], TensorType] = None
    sign: Callable[[TensorType], TensorType] = None
    create_rng: Callable[[int], RandomGeneratorType] = None
    standard_normal: Callable[[RandomGeneratorType, "tuple[int]"], TensorType] = None
    uniform: Callable[
        [RandomGeneratorType, "tuple[int]", TensorType, TensorType, type], TensorType
    ] = None
    sum: Callable[[TensorType, "Optional[Union[tuple[int], int]]"], TensorType] = None
    mean: Callable[[TensorType, "Optional[Union[tuple[int], int]]"], TensorType] = None
    cumsum: Callable[[TensorType, int], TensorType] = None
    cumprod: Callable[[TensorType, int], TensorType] = None
    set_shape: Callable[[TensorType, "list[int]"], None] = None
    concat: Callable[["list[TensorType, ...]", int], TensorType]
    pi: TensorType = None
    any: Callable[[TensorType], bool] = None
    all: Callable[[TensorType], bool] = None
    reduce_any: Callable[[TensorType, int], bool] = None
    reduce_all: Callable[[TensorType, int], bool] = None
    reduce_max: Callable[[TensorType, int], bool] = None
    reduce_min: Callable[[TensorType, Optional[int]], bool] = None
    equal: Callable[[TensorType, TensorType], TensorType] = None
    less: Callable[[TensorType, TensorType], TensorType] = None
    less_equal: Callable[[TensorType, TensorType], TensorType] = None
    greater: Callable[[TensorType, TensorType], TensorType] = None
    greater_equal: Callable[[TensorType, TensorType], TensorType] = None
    logical_not: Callable[[TensorType], TensorType] = None
    min: Callable[[TensorType, TensorType], TensorType] = None
    max: Callable[[TensorType, TensorType], TensorType] = None
    atan: Callable[[TensorType], TensorType] = None
    atan2: Callable[[TensorType], TensorType] = None
    abs: Callable[[TensorType], TensorType] = None
    sqrt: Callable[[TensorType], TensorType] = None
    argpartition: Callable[[TensorType, int], TensorType] = None
    argmax: Callable[[TensorType, int], TensorType] = None
    norm: Callable[[TensorType, int], bool] = None
    matmul: Callable[[TensorType, TensorType], TensorType] = None
    cross: Callable[[TensorType, TensorType], TensorType] = None
    dot: Callable[[TensorType, TensorType], TensorType] = None
    stop_gradient: Callable[[TensorType], TensorType] = None
    assign: Callable[[Union[TensorType, "tensorflow.Variable"], TensorType], Union[TensorType, "tensorflow.Variable"]] = None
    where: Callable[[TensorType, TensorType, TensorType], TensorType] = None
    cond: Callable[[TensorType, Callable[[], Any], Callable[[], Any]], TensorType] = None
    logical_and: Callable[[TensorType, TensorType], TensorType] = None
    logical_or: Callable[[TensorType, TensorType], TensorType] = None
    print: Callable[[Any], None] = None
    square: Callable[[TensorType], TensorType] = None
    divide: Callable[[TensorType, TensorType], TensorType] = None


class NumpyLibrary(ComputationLibrary):

    def __init__(self):
        import numpy as np  # Lazy import here

        self.lib = 'Numpy'
        self.float32 = np.float32
        self.float64 = np.float64
        self.int32 = np.int32
        self.int64 = np.int64
        
        self.lib = 'Numpy'
        self.set_device = lambda device_name: set_device_general(device_name, 'numpy')
        self.reshape = lambda x, shape: np.reshape(x, shape)
        self.permute = np.transpose
        self.newaxis = np.newaxis
        self.shape = np.shape
        self.to_numpy = lambda x: np.array(x)
        self.to_variable = lambda x, dtype: np.array(x, dtype=dtype)
        self.to_tensor = lambda x, dtype: np.array(x, dtype=dtype)
        self.constant = lambda x, t: np.array(x, dtype=t)
        self.unstack = lambda x, num, axis: list(np.moveaxis(x, axis, 0))
        self.ndim = np.ndim
        self.clip = np.clip
        self.sin = np.sin
        self.asin = np.arcsin
        self.cos = np.cos
        self.tan = np.tan
        self.tanh = np.tanh
        self.exp = np.exp
        self.reciprocal = np.reciprocal
        self.squeeze = np.squeeze
        self.unsqueeze = np.expand_dims
        self.stack = np.stack
        self.cast = lambda x, t: x.astype(t)
        self.floormod = np.mod
        self.floor = np.floor
        self.ceil = np.ceil
        self.rint = np.rint
        self.float32 = np.float32
        self.float64 = np.float64
        self.int32 = np.int32
        self.int64 = np.int64
        self.bool = np.bool_
        self.tile = np.tile
        self.repeat = lambda x, k, a: np.repeat(x, repeats=k, axis=a)
        self.gather = lambda x, i, a: np.take(x, i, axis=a)
        self.gather_last = lambda x, i: np.take(x, i, axis=-1)
        self.arange = np.arange
        self.zeros = np.zeros
        self.zeros_like = np.zeros_like
        self.ones = np.ones
        self.ones_like = np.ones_like
        self.sign = np.sign
        self.create_rng = lambda seed: np.random.Generator(np.random.SFC64(seed))
        self.standard_normal = lambda generator, shape: generator.standard_normal(size=shape)
        self.uniform = lambda generator, shape, low, high, dtype: generator.uniform(
            low=low, high=high, size=shape
        ).astype(dtype)
        self.sum = lambda x, a: np.sum(x, axis=a, keepdims=False)
        self.mean = lambda x, a: np.mean(x, axis=a, keepdims=False)
        self.cumsum = lambda x, a: np.cumsum(x, axis=a)
        self.cumprod = lambda x, a: np.cumprod(x, axis=a)
        self.set_shape = lambda x, shape: x
        self.concat = lambda x, axis: np.concatenate(x, axis=axis)
        self.pi = np.array(np.pi).astype(np.float32)
        self.any = np.any
        self.all = np.all
        self.reduce_any = lambda a, axis: np.any(a, axis=axis)
        self.reduce_all = lambda a, axis: np.all(a, axis=axis)
        self.reduce_max = lambda a, axis: np.max(a, axis=axis)
        self.reduce_min = lambda a, axis: np.min(a, axis=axis)
        self.equal = lambda x, y: np.equal(x, y)
        self.less = lambda x, y: np.less(x, y)
        self.less_equal = lambda x, y: np.less_equal(x, y)
        self.greater = lambda x, y: np.greater(x, y)
        self.greater_equal = lambda x, y: np.greater_equal(x, y)
        self.logical_not = lambda x: np.logical_not(x)
        self.min = np.minimum
        self.max = np.maximum
        self.atan = np.arctan
        self.atan2 = np.arctan2
        self.abs = np.abs
        self.sqrt = np.sqrt
        self.argpartition = lambda x, k: np.argpartition(x, k)[..., :k]
        self.argmax = lambda x, a: np.argmax(x, axis=a)
        self.norm = lambda x, axis: np.linalg.norm(x, axis=axis)
        self.matmul = np.matmul
        self.cross = np.cross
        self.dot = np.dot
        self.stop_gradient = lambda x: x
        self.where = np.where
        self.logical_and = np.logical_and
        self.logical_or  = np.logical_or
        self.print = print
        self.square = np.square
        self.divide = np.divide

    @staticmethod
    def assign(v: "TensorType", x: "TensorType"):
        import numpy as np
        if isinstance(v, np.ndarray):
            v[...] = x
        else:
            raise TypeError("Unsupported tensor type")

    @staticmethod
    def cond(condition: "TensorType", true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> Any:
        import numpy as np
        if isinstance(condition, np.ndarray):
            return true_fn() if condition else false_fn()
        else:
            raise TypeError("Unsupported tensor type")


class TensorFlowLibrary(ComputationLibrary):
    def __init__(self):
        import tensorflow as tf  # Lazy import
        self.lib = 'TF'
        self.float32 = tf.float32
        self.float64 = tf.float64

    
        self.set_device = lambda device_name: set_device_general(device_name, 'tf')
        self.reshape = tf.reshape
        self.permute = tf.transpose
        self.newaxis = tf.newaxis
        self.shape = tf.shape
        self.to_numpy = lambda x: x.numpy()
        self.to_variable = lambda x, dtype: tf.Variable(x, dtype=dtype)
        self.to_tensor = lambda x, dtype: tf.convert_to_tensor(x, dtype=dtype)
        self.constant = lambda x, t: tf.constant(x, dtype=t)
        self.unstack = lambda x, num, axis: tf.unstack(x, num=num, axis=axis)
        self.ndim = tf.rank
        self.clip = tf.clip_by_value
        self.sin = tf.sin
        self.asin = tf.asin
        self.cos = tf.cos
        self.tan = tf.tan
        self.tanh = tf.tanh
        self.exp = tf.exp
        self.reciprocal = tf.math.reciprocal
        self.squeeze = tf.squeeze
        self.unsqueeze = tf.expand_dims
        self.stack = tf.stack
        self.cast = lambda x, t: tf.cast(x, dtype=t)
        self.floormod = tf.math.floormod
        self.floor = tf.math.floor
        self.ceil = tf.math.ceil
        self.rint = tf.math.rint
        self.float32 = tf.float32
        self.float64 = tf.float64
        self.int32 = tf.int32
        self.int64 = tf.int64
        self.bool = tf.bool
        self.tile = tf.tile
        self.repeat = lambda x, k, a: tf.repeat(x, repeats=k, axis=a)
        self.gather = lambda x, i, a: tf.gather(x, i, axis=a)
        self.gather_last = lambda x, i: tf.gather(x, i, axis=-1)
        self.arange = tf.range
        self.zeros = tf.zeros
        self.zeros_like = tf.zeros_like
        self.ones = tf.ones
        self.ones_like = tf.ones_like
        self.sign = tf.sign
        self.create_rng = lambda seed: tf.random.Generator.from_seed(seed)
        self.standard_normal = lambda generator, shape: generator.normal(shape)
        self.uniform = lambda generator, shape, low, high, dtype: generator.uniform(
            shape, minval=low, maxval=high, dtype=dtype)
        self.sum = lambda x, a: tf.reduce_sum(x, axis=a, keepdims=False)
        self.mean = lambda x, a: tf.reduce_mean(x, axis=a, keepdims=False)
        self.cumsum = lambda x, a: tf.math.cumsum(x, axis=a)
        self.cumprod = lambda x, a: tf.math.cumprod(x, axis=a)
        self.set_shape = lambda x, shape: x.set_shape(shape)
        self.concat = lambda x, axis: tf.concat(x, axis)
        self.pi = tf.convert_to_tensor(pi, dtype=tf.float32)
        self.any = tf.reduce_any
        self.all = tf.reduce_all
        self.reduce_any = lambda a, axis: tf.reduce_any(a, axis=axis)
        self.reduce_all = lambda a, axis: tf.reduce_all(a, axis=axis)
        self.reduce_max = lambda a, axis: tf.reduce_max(a, axis=axis)
        self.reduce_min = lambda a, axis: tf.reduce_min(a, axis=axis)
        self.equal = lambda x, y: tf.math.equal(x, y)
        self.less = lambda x, y: tf.math.less(x, y)
        self.less_equal = lambda x, y: tf.math.less_equal(x, y)
        self.greater = lambda x, y: tf.math.greater(x, y)
        self.greater_equal = lambda x, y: tf.math.greater_equal(x, y)
        self.logical_not = lambda x: tf.math.logical_not(x)
        self.min = tf.minimum
        self.max = tf.maximum
        self.atan = tf.math.atan
        self.atan2 = tf.atan2
        self.abs = tf.abs
        self.sqrt = tf.sqrt
        self.argpartition = lambda x, k: tf.math.top_k(-x, k, sorted=False)[1]
        self.argmax = lambda x, a: tf.math.argmax(x, axis=a)
        self.norm = lambda x, axis: tf.norm(x, axis=axis)
        self.matmul = tf.linalg.matmul
        self.cross = tf.linalg.cross
        self.dot = lambda a, b: tf.tensordot(a, b, 1)
        self.stop_gradient = tf.stop_gradient
        self.where = tf.where
        self.cond = tf.cond
        self.logical_and = tf.math.logical_and
        self.logical_or  = tf.math.logical_or
        self.print = tf.print
        self.square = tf.square
        self.divide = tf.math.divide

    @staticmethod
    def assign(v: "TensorType", x: "TensorType"):
        import tensorflow as tf  # Lazy import
        if isinstance(v, tf.Variable):
            v.assign(x)
        else:
            raise TypeError("Unsupported tensor type")


class PyTorchLibrary(ComputationLibrary):
    
    def __init__(self):
        import torch  # Lazy import here
        self.lib = 'Pytorch'
        self.float32 = torch.float32
        self.float64 = torch.float64
        
        
        self.lib = 'Pytorch'
        self.set_device = lambda device_name: set_device_general(device_name, 'pytorch')
        self.reshape = torch.reshape
        self.permute = torch.permute
        self.newaxis = None
        self.shape = lambda x: list(x.size())
        self.to_numpy = lambda x: x.cpu().detach().numpy()
        self.to_variable = lambda x, dtype: torch.as_tensor(x, dtype=dtype)
        self.to_tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype)
        self.constant = lambda x, t: torch.as_tensor(x, dtype=t)
        self.unstack = lambda x, num, dim: torch.unbind(x, dim=dim)
        self.ndim = lambda x: x.ndim
        self.clip = torch.clamp
        self.sin = torch.sin
        self.asin = torch.asin
        self.cos = torch.cos
        self.tan = torch.tan
        self.tanh = torch.tanh
        self.exp = torch.exp
        self.reciprocal = torch.reciprocal
        self.squeeze = torch.squeeze
        self.unsqueeze = torch.unsqueeze
        self.stack = torch.stack
        self.cast = lambda x, t: x.type(t)
        self.floormod = torch.remainder
        self.floor = lambda x: torch.floor(torch.as_tensor(x))
        self.ceil = lambda x: torch.ceil(torch.as_tensor(x))
        self.rint = lambda x: torch.round(torch.as_tensor(x), decimals=0)
        self.float32 = torch.float32
        self.float64 = torch.float64
        self.int32 = torch.int32
        self.int64 = torch.int64
        self.bool = torch.bool
        self.tile = torch.tile
        self.repeat = lambda x, k, a: torch.repeat_interleave(x, repeats=k, dim=a)
        self.gather = lambda x, i, a: torch.gather(x, dim=a, index=i)  # FIXME: It works very differently to TF!!!
        self.gather_last = self.gather_last_pytorch
        self.arange = torch.arange
        self.zeros = torch.zeros
        self.zeros_like = torch.zeros_like
        self.ones = torch.ones
        self.ones_like = torch.ones_like
        self.sign = torch.sign
        self.create_rng = lambda seed: torch.Generator().manual_seed(seed)
        self.standard_normal = lambda generator, shape: torch.normal(
            torch.zeros(shape), 1.0, generator=generator
        )
        self.uniform = (
            lambda generator, shape, low, high, dtype: (high - low)
            * torch.rand(*shape, generator=generator, dtype=dtype)
            + low
        )
        self.sum = lambda x, a: torch.sum(x, a, keepdim=False)
        self.mean = lambda x, a: torch.mean(x, a, keepdim=False)
        self.cumsum = lambda x, a: torch.cumsum(x, dim=a)
        self.cumprod = lambda x, a: torch.cumprod(x, dim=a)
        self.set_shape = lambda x, shape: x
        self.concat = lambda x, axis: torch.concat(x, dim=axis)
        self.pi = torch.as_tensor(pi, dtype=torch.float32)
        self.any = torch.any
        self.all = torch.all
        self.reduce_any = lambda a, axis: torch.any(a, dim=axis)
        self.reduce_all = lambda a, axis: torch.all(a, dim=axis)
        self.reduce_max = lambda a, axis: torch.max(a, dim=axis)
        self.reduce_min = lambda a, axis: torch.min(a, dim=axis)[0]
        self.equal = lambda x, y: torch.eq(x, y)
        self.less = lambda x, y: torch.less(x, y)
        self.less_equal = lambda x, y: torch.less_equal(x, y)
        self.greater = lambda x, y: torch.greater(x, y)
        self.greater_equal = lambda x, y: torch.greater_equal(x, y)
        self.logical_not = lambda x: torch.logical_not(x)
        self.min = torch.minimum
        self.max = torch.maximum
        self.atan = torch.atan
        self.atan2 = torch.atan2
        self.abs = torch.abs
        self.sqrt = torch.sqrt
        self.argpartition = torch.topk
        self.argmax = lambda x, a: torch.argmax(x, dim=a)
        self.norm = lambda x, axis: torch.linalg.norm(x, dim=axis)
        self.matmul = torch.matmul
        self.cross = torch.linalg.cross
        self.dot = torch.dot
        # stop_gradient = tf.stop_gradient # FIXME: How to imlement this in torch?
        self.where = torch.where
        self.logical_and = torch.logical_and
        self.logical_or  = torch.logical_or
        self.print = print
        self.square = torch.square
        self.divide = torch.div

    @staticmethod
    def assign(v: "TensorType", x: "TensorType"):
        import torch
        if isinstance(v, torch.Tensor):
            v.copy_(x)
        else:
            raise TypeError("Unsupported tensor type")

    @staticmethod
    def cond(condition: "TensorType", true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> Any:
        import torch
        if isinstance(condition, torch.Tensor):
            return true_fn() if condition.item() else false_fn()
        else:
            raise TypeError("Unsupported tensor type")


    @staticmethod
    def gather_last_pytorch(a, index_vector):
        return a[..., index_vector]


ComputationClasses = (NumpyLibrary, TensorFlowLibrary, PyTorchLibrary)