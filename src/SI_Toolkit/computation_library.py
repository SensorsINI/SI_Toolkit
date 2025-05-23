from typing import Callable, Optional, Union, Sequence, Any

import functools

from math import pi


# TensorType = Union[np.ndarray, tf.Tensor, torch.Tensor]
TensorType = Union["np.ndarray", "tf.Tensor", "torch.Tensor"]
VariableType = Union["np.ndarray", "tf.Variable", "torch.Tensor"]
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
        # We accept either '/cpu:0' style or '/physical_cpu:0' style names
        physical_names = [d.name for d in devices]
        alias = '/physical_' + device_name[1:]
        if device_name not in physical_names and alias not in physical_names:
            raise ValueError(
                f"Requested device {device_name!r} not found among physical devices: {physical_names}"
            )

        # TensorFlow device management
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with tf.device(device_name):
                    return func(*args, **kwargs)
            return wrapper

    elif library == 'pytorch':

        import torch

        # ─── Normalize various device-string formats ──────────────────────────────
        # TensorFlow emits '/device:TYPE:IDX'; shorthand '/gpu:0'; convert both to PyTorch style.
        dev_str = device_name

        if dev_str.lower().startswith('/device:'):
            # '/device:GPU:1' → ['device','GPU','1']
            _, type_str, idx = dev_str.split(':', 2)
            type_str = type_str.lower()
            dev_str = 'cpu' if type_str == 'cpu' else f'cuda:{idx}'

        elif dev_str.startswith('/'):
            # '/gpu:0' → ['gpu','0']
            type_str, idx = dev_str[1:].split(':', 1)
            type_str = type_str.lower()
            dev_str = 'cpu' if type_str == 'cpu' else f'cuda:{idx}'
        # else assume already 'cpu' or 'cuda:0'

        # ─── Enumerate available devices ─────────────────────────────────────────────
        devices = [torch.device('cpu')]  # always allow CPU
        if torch.cuda.is_available():
            # include every GPU index
            devices += [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

        target_device = torch.device(dev_str)
        if target_device not in devices:
            raise ValueError(
                f"Requested device {device_name!r} (normalized to {dev_str!r}) "
                f"not found among available devices: {devices}"
            )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # If first argument is a module/tensor with .to(), migrate it eagerly
                if args and hasattr(args[0], 'to'):
                    first = args[0]
                    if getattr(first, 'device', None) != target_device:
                        first = first.to(target_device)
                    args = (first,) + args[1:]

                # Then, handle all the args and kwargs
                new_args = [
                    arg.to(target_device)
                    if isinstance(arg, torch.Tensor) and arg.device != target_device
                    else arg
                    for arg in args
                ]
                new_kwargs = {
                    k: (v.to(target_device)
                        if isinstance(v, torch.Tensor) and v.device != target_device
                        else v)
                    for k, v in kwargs.items()
                }
                return func(*new_args, **new_kwargs)

            return wrapper
    else:
        raise ValueError(f'Invalid library name: {library!r}')

    return decorator


def clip_by_norm_factory(lib):

    def clip_by_norm(x, clip_norm, axis=(-1,), eps=1e-9):
        norm   = lib.norm(x, axis=axis, keepdims=True)
        normed_x = clip_norm / (norm + eps)
        factor = lib.min(lib.to_tensor(1.0, lib.float32), normed_x)
        return x * factor

    return clip_by_norm


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
    pad: Callable[[TensorType, Any], TensorType] = None
    clip: Callable[[TensorType, float, float], TensorType] = None
    clip_by_norm: Callable[[TensorType, TensorType, Union[int, Sequence[int]]], TensorType] = None
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
    zeros: Callable[..., TensorType] = None
    zeros_like: Callable[[TensorType], TensorType] = None
    ones: Callable[["tuple[int]"], TensorType] = None
    ones_like: Callable[[TensorType], TensorType] = None
    sign: Callable[[TensorType], TensorType] = None
    create_rng: Callable[[int], RandomGeneratorType] = None
    standard_normal: Callable[[RandomGeneratorType, "tuple[int]"], TensorType] = None
    uniform: Callable[
        [RandomGeneratorType, "tuple[int]", TensorType, TensorType, type], TensorType
    ] = None
    sum: Callable[..., TensorType] = None
    mean: Callable[[TensorType, "Optional[Union[tuple[int], int]]"], TensorType] = None
    cumsum: Callable[[TensorType, int], TensorType] = None
    cumprod: Callable[[TensorType, int], TensorType] = None
    set_shape: Callable[[TensorType, "list[int]"], None] = None
    concat: Callable[[Sequence[TensorType], int], TensorType] = None
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
    argsort: Callable[[TensorType, int], TensorType] = None
    argpartition: Callable[[TensorType, int], TensorType] = None
    argmin: Callable[[TensorType, int], TensorType] = None
    argmax: Callable[[TensorType, int], TensorType] = None
    norm: Callable[[TensorType, int, bool], TensorType] = None
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
    multiply: Callable[[TensorType, TensorType], TensorType] = None
    divide: Callable[[TensorType, TensorType], TensorType] = None
    subtract: Callable[[TensorType, TensorType], TensorType] = None
    GradientTape = None
    break_compilation_graph = None


class NumpyLibrary(ComputationLibrary):

    def __init__(self):
        import numpy as np  # Lazy import here

        self.lib = 'Numpy'
        self.float32 = np.float32
        self.float64 = np.float64
        self.int32 = np.int32
        self.int64 = np.int64
        self.set_device = lambda device_name: set_device_general(device_name, 'numpy')
        self.reshape = lambda x, shape: np.reshape(x, shape)
        self.permute = np.transpose
        self.newaxis = np.newaxis
        self.shape = np.shape
        self.to_numpy = lambda x: np.array(x)
        self.to_variable = lambda x, dtype: np.array(x, dtype=dtype)
        self.to_tensor = lambda x, dtype: np.array(x, dtype=dtype)
        self.constant = lambda x, t: np.array(x, dtype=t)
        self.unstack = lambda x, num, axis: [np.take(x, i, axis=axis) for i in range(num)]
        self.ndim = np.ndim
        self.pad = lambda x, paddings: np.pad(x, paddings, mode='constant')
        self.clip = np.clip
        self.clip_by_norm = clip_by_norm_factory(self)
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
        self.gather = lambda x, i, axis: np.take(x, i, axis=axis)
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
        self.sum = lambda x, axis: np.sum(x, axis=axis, keepdims=False)
        self.mean = lambda x, axis: np.mean(x, axis=axis, keepdims=False)
        self.cumsum = lambda x, axis: np.cumsum(x, axis=axis)
        self.cumprod = lambda x, axis: np.cumprod(x, axis=axis)
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
        self.argsort = lambda x, axis=-1: np.argsort(x, axis=axis)
        self.argpartition = lambda x, k: np.argpartition(x, k)[..., :k]
        self.argmin = lambda x, axis: np.argmin(x, axis=axis)
        self.argmax = lambda x, a: np.argmax(x, axis=a)
        self.norm = lambda x, axis, keepdims=False: np.linalg.norm(x, axis=axis, keepdims=keepdims)
        self.matmul = np.matmul
        self.cross = np.cross
        self.dot = np.dot
        self.stop_gradient = lambda x: x
        self.where = np.where
        self.logical_and = np.logical_and
        self.logical_or  = np.logical_or
        self.print = print
        self.square = np.square
        self.multiply = np.multiply
        self.divide = np.divide
        self.subtract = np.subtract
        self.break_compilation_graph = lambda: None

    @staticmethod
    def loop(body_fn, state, steps: int, counter: int = 0):
        """
        Run `body_fn` `steps` times starting from `state` and
        return the final state.

        * `body_fn(*state)` → new_state  (same structure)
        * Works for TF, PyTorch, NumPy.
        """

        # ──────❶ flatten the incoming state tuple ────────────────
        # so that state = (counter, *state)
        state = (counter, *state)

        for _ in range(steps):
            state = body_fn(*state)

        return state

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
        self.pad = tf.pad
        self.clip_by_norm = clip_by_norm_factory(self)
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
        self.gather = lambda x, i, axis: tf.gather(x, i, axis=axis)
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
        self.sum = lambda x, axis: tf.reduce_sum(x, axis=axis, keepdims=False)
        self.mean = lambda x, axis: tf.reduce_mean(x, axis=axis, keepdims=False)
        self.cumsum = lambda x, axis: tf.math.cumsum(x, axis=axis)
        self.cumprod = lambda x, axis: tf.math.cumprod(x, axis=axis)
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
        self.argsort = lambda x, axis=-1: tf.argsort(x, axis=axis)
        self.argpartition = lambda x, k: tf.math.top_k(-x, k, sorted=False)[1]
        self.argmin = lambda x, axis: tf.math.argmin(x, axis=axis)
        self.argmax = lambda x, a: tf.math.argmax(x, axis=a)
        self.norm = lambda x, axis, keepdims=False: tf.norm(x, axis=axis, keepdims=keepdims)
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
        self.multiply = tf.multiply
        self.divide = tf.math.divide
        self.subtract = tf.math.subtract
        self.GradientTape = tf.GradientTape
        self.break_compilation_graph = lambda: None

    @staticmethod
    def loop(body_fn,
             state,
             steps,
             counter = 0):
        """
        XLA-friendly loop.
        Returns (counter, *state) exactly as your older helper did,
        so existing call-sites remain valid.
        """
        import tensorflow as tf

        # ─── normalise inputs once ─────────────────────────────────────────
        steps = tf.convert_to_tensor(steps, tf.int32, name="steps")
        counter = tf.convert_to_tensor(counter, tf.int32, name="counter")

        # loop variables = (iteration index, logical counter, *state)
        loop_vars = (tf.constant(0, tf.int32),
                     counter,
                     *state)

        def cond(i, *_):
            return i < steps

        def body(i, c, *s):
            # delegate to user fn; it must return (new_c, *new_state)
            c, *s = body_fn(c, *s)
            # be lenient: tensorise once here, not every turn at outer level
            c = tf.convert_to_tensor(c, tf.int32, name="counter_out")
            return (i + 1, c, *s)

        _, c_final, *state_final = tf.while_loop(
            cond, body, loop_vars,
            parallel_iterations=32,
            maximum_iterations=steps
        )

        # public contract preserved
        return (c_final, *state_final)

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
        import torch.nn.functional as F

        self.lib = 'Pytorch'
        self.float32 = torch.float32
        self.float64 = torch.float64
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
        self.pad = lambda x, paddings: F.pad(x, [p for dim in reversed(paddings) for p in reversed(dim)], mode='constant', value=0.0)
        self.clip_by_norm = clip_by_norm_factory(self)
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
        self.gather = tf_gather_for_torch
        self.gather_last = self.gather_last_pytorch
        self.arange = torch.arange
        self.zeros = lambda shape, **kw: torch.zeros(size=shape, **kw)
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
        self.sum = lambda x, axis: torch.sum(x, axis, keepdim=False)
        self.mean = lambda x, axis: torch.mean(x, axis, keepdim=False)
        self.cumsum = lambda x, axis: torch.cumsum(x, dim=axis)
        self.cumprod = lambda x, axis: torch.cumprod(x, dim=axis)
        self.set_shape = lambda x, shape: x
        self.concat = lambda x, axis: torch.concat(x, dim=axis)
        self.pi = torch.as_tensor(pi, dtype=torch.float32)
        self.any = torch.any
        self.all = torch.all
        self.reduce_any = lambda a, axis: torch.any(a, dim=axis)
        self.reduce_all = lambda a, axis: torch.all(a, dim=axis)
        self.reduce_max = lambda a, axis: torch.max(a, dim=axis)[0]
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
        self.argsort = lambda x, axis=-1: torch.argsort(x, dim=axis)
        self.argpartition = lambda x, k: torch.topk(x, k, largest=False)[1]
        self.argmin = lambda x, axis: torch.argmin(x, dim=axis)
        self.argmax = lambda x, a: torch.argmax(x, dim=a)
        self.norm = lambda x, axis, keepdims=False: torch.linalg.norm(x, dim=axis, keepdim=keepdims)
        self.matmul = torch.matmul
        self.cross = torch.linalg.cross
        self.dot = torch.dot
        self.stop_gradient = lambda x: x.detach()
        self.where = torch.where
        self.logical_and = torch.logical_and
        self.logical_or  = torch.logical_or
        self.print = print
        self.square = torch.square
        self.multiply = torch.mul
        self.divide = torch.div
        self.subtract = torch.subtract
        self.break_compilation_graph = pytorch_break_compilation_graph

    def loop(self, body_fn, state, steps: int, counter = 0):
        """
        Run `body_fn` `steps` times starting from `state` and return the final state.

        * `body_fn(*state)` → new_state (same structure)
        * Works for TF, PyTorch, NumPy.
        """

        import torch

        # ────❶ normalise counter once, *outside* the traced loop ────────
        counter = torch.as_tensor(counter, dtype=torch.int64, device=state[0].device)

        # ─── flatten the incoming state tuple ───
        state = (counter, *state)

        # ─── pure-Python loop; will be unrolled when `steps` is constant ───
        for _ in range(steps):
            self.break_compilation_graph()
            # NOTE: no graph break here; keep the loop in one FX sub-graph
            state = body_fn(*state)

            # ❷ Enforce invariant: `state[0]` *must* stay a tensor.
            #    Instead of branching, assume well-behaved `body_fn` and fail
            #    loudly if it violates the contract.  This avoids dynamic
            #    recompilation guards.
            assert torch.is_tensor(state[0]), "body_fn must return a tensor counter"

        # ─── optional: a single break *after* the loop if you need to mix
        #     non-traceable Python with the result; remove otherwise
        self.break_compilation_graph()

        return state

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
        import torch
        # index_vector: (...,) long tensor indexing the last dim of `a`
        idx = index_vector.unsqueeze(-1)  # now shape (...,1)
        out = torch.gather(a, dim=-1, index=idx)  # shape (...,1)
        return out.squeeze(-1)


ComputationClasses = (NumpyLibrary, TensorFlowLibrary, PyTorchLibrary)


def tf_gather_for_torch(input, index, axis=0):
    """
    PyTorch equivalent of tf.gather(input, index, axis=axis).
    Accepts lists or NumPy arrays as well as Tensors.
    """
    import torch
    dim = axis

    # ensure our data is a Tensor, without copying if it already is one
    if not torch.is_tensor(input):
        input = torch.as_tensor(input)
    # make sure indices are a LongTensor on the same device as `input`
    if not torch.is_tensor(index):
        index = torch.as_tensor(index, dtype=torch.long, device=input.device)
    else:
        if index.dtype != torch.long:
            index = index.to(torch.long)
        if index.device != input.device:
            index = index.to(input.device)

    # PyTorch’s index_select only accepts 1-D index tensors.
    # For N-D `index`, we flatten → select → reshape back.
    if index.dim() > 1:
        # flatten all index dimensions into one vector
        flat_idx = index.reshape(-1)
        # gather the rows/slices
        gathered = torch.index_select(input, dim, flat_idx)
        # compute the output shape: original index shape + remaining dims of input after `dim`
        out_shape = list(index.shape) + list(input.shape[dim + 1 :])
        return gathered.reshape(*out_shape)
    else:
        # simple 1-D case: direct select
        return torch.index_select(input, dim, index)


def pytorch_break_compilation_graph():
    """
    Allows splitting the compilation graph in PyTorch. To make compilation faster/feasible.
    """
    import torch._dynamo as dynamo
    dynamo.graph_break()

