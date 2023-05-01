from typing import Callable, Optional, Union, Sequence

import numpy as np
import tensorflow as tf
import torch
from numpy.random import Generator, SFC64

TensorType = Union[np.ndarray, tf.Tensor, torch.Tensor]
RandomGeneratorType = Union[Generator, tf.random.Generator, torch.Generator]
NumericType = Union[float, int]


class LibraryHelperFunctions:
    @staticmethod
    def set_to_value(v: TensorType, x: TensorType):
        v[...] = x

    @staticmethod
    def set_to_variable(v: tf.Variable, x: tf.Tensor):
        v.assign(x)


class ComputationLibrary:
    lib = None
    reshape: Callable[[TensorType, "tuple[int]"], TensorType] = None
    permute: Callable[[TensorType, "tuple[int]"], TensorType] = None
    newaxis = None
    shape: Callable[[TensorType], "list[int]"] = None
    to_numpy: Callable[[TensorType], np.ndarray] = None
    to_variable: Callable[[TensorType, type], np.ndarray] = None
    to_tensor: Callable[[TensorType, type], TensorType] = None
    constant: Callable[[TensorType, type], TensorType] = None
    unstack: Callable[[TensorType, int, int], "list[TensorType]"] = None
    ndim: Callable[[TensorType], int] = None
    clip: Callable[[TensorType, float, float], TensorType] = None
    sin: Callable[[TensorType], TensorType] = None
    asin: Callable[[TensorType], TensorType] = None
    cos: Callable[[TensorType], TensorType] = None
    tan: Callable[[TensorType], TensorType] = None
    tanh: Callable[[TensorType], TensorType] = None
    exp: Callable[[TensorType], TensorType] = None
    squeeze: Callable[[TensorType], TensorType] = None
    unsqueeze: Callable[[TensorType, int], TensorType] = None
    stack: Callable[["list[TensorType]", int], TensorType] = None
    cast: Callable[[TensorType, type], TensorType] = None
    floormod: Callable[[TensorType], TensorType] = None
    floor: Callable[[TensorType], TensorType] = None
    ceil: Callable[[TensorType], TensorType] = None
    float32 = None
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
    concat: Callable[["list[TensorType]", int], TensorType]
    pi: TensorType = None
    any: Callable[[TensorType], bool] = None
    all: Callable[[TensorType], bool] = None
    reduce_any: Callable[[TensorType, int], bool] = None
    reduce_all: Callable[[TensorType, int], bool] = None
    reduce_max: Callable[[TensorType, int], bool] = None
    reduce_min: Callable[[TensorType, Optional[int]], bool] = None
    less: Callable[[TensorType, TensorType], TensorType] = None
    less_equal: Callable[[TensorType, TensorType], TensorType] = None
    greater: Callable[[TensorType, TensorType], TensorType] = None
    greater_equal: Callable[[TensorType, TensorType], TensorType] = None
    logical_not: Callable[[TensorType], TensorType] = None
    min: Callable[[TensorType, TensorType], TensorType] = None
    max: Callable[[TensorType, TensorType], TensorType] = None
    atan2: Callable[[TensorType], TensorType] = None
    abs: Callable[[TensorType], TensorType] = None
    sqrt: Callable[[TensorType], TensorType] = None
    argpartition: Callable[[TensorType, int], TensorType] = None
    norm: Callable[[TensorType, int], bool] = None
    matmul: Callable[[TensorType, TensorType], TensorType] = None
    cross: Callable[[TensorType, TensorType], TensorType] = None
    dot: Callable[[TensorType, TensorType], TensorType] = None
    stop_gradient: Callable[[TensorType], TensorType] = None
    assign: Callable[[Union[TensorType, tf.Variable], TensorType], Union[TensorType, tf.Variable]] = None
    where: Callable[[TensorType, TensorType, TensorType], TensorType] = None
    logical_and: Callable[[TensorType, TensorType], TensorType] = None
    logical_or: Callable[[TensorType, TensorType], TensorType] = None


class NumpyLibrary(ComputationLibrary):
    lib = 'Numpy'
    reshape = lambda x, shape: np.reshape(x, shape)
    permute = np.transpose
    newaxis = np.newaxis
    shape = np.shape
    to_numpy = lambda x: np.array(x)
    to_variable = lambda x, dtype: np.array(x, dtype=dtype)
    to_tensor = lambda x, dtype: np.array(x, dtype=dtype)
    constant = lambda x, t: np.array(x, dtype=t)
    unstack = lambda x, num, axis: list(np.moveaxis(x, axis, 0))
    ndim = np.ndim
    clip = np.clip
    sin = np.sin
    asin = np.arcsin
    cos = np.cos
    tan = np.tan
    tanh = np.tanh
    exp = np.exp
    squeeze = np.squeeze
    unsqueeze = np.expand_dims
    stack = np.stack
    cast = lambda x, t: x.astype(t)
    floormod = np.mod
    floor = np.floor
    ceil = np.ceil
    float32 = np.float32
    int32 = np.int32
    int64 = np.int64
    bool = np.bool_
    tile = np.tile
    repeat = lambda x, k, a: np.repeat(x, repeats=k, axis=a)
    gather = lambda x, i, a: np.take(x, i, axis=a)
    gather_last = lambda x, i: np.take(x, i, axis=-1)
    arange = np.arange
    zeros = np.zeros
    zeros_like = np.zeros_like
    ones = np.ones
    ones_like = np.ones_like
    sign = np.sign
    create_rng = lambda seed: Generator(SFC64(seed))
    standard_normal = lambda generator, shape: generator.standard_normal(size=shape)
    uniform = lambda generator, shape, low, high, dtype: generator.uniform(
        low=low, high=high, size=shape
    ).astype(dtype)
    sum = lambda x, a: np.sum(x, axis=a, keepdims=False)
    mean = lambda x, a: np.mean(x, axis=a, keepdims=False)
    cumsum = lambda x, a: np.cumsum(x, axis=a)
    cumprod = lambda x, a: np.cumprod(x, axis=a)
    set_shape = lambda x, shape: x
    concat = lambda x, axis: np.concatenate(x, axis=axis)
    pi = np.array(np.pi).astype(np.float32)
    any = np.any
    all = np.all
    reduce_any = lambda a, axis: np.any(a, axis=axis)
    reduce_all = lambda a, axis: np.all(a, axis=axis)
    reduce_max = lambda a, axis: np.max(a, axis=axis)
    reduce_min = lambda a, axis: np.min(a, axis=axis)
    less = lambda x, y: np.less(x, y)
    less_equal = lambda x, y: np.less_equal(x, y)
    greater = lambda x, y: np.greater(x, y)
    greater_equal = lambda x, y: np.greater_equal(x, y)
    logical_not = lambda x: np.logical_not(x)
    min = np.minimum
    max = np.maximum
    atan2 = np.arctan2
    abs = np.abs
    sqrt = np.sqrt
    argpartition = lambda x, k: np.argpartition(x, k)[..., :k]
    norm = lambda x, axis: np.linalg.norm(x, axis=axis)
    matmul = np.matmul
    cross = np.cross
    dot = np.dot
    stop_gradient = lambda x: x
    assign = LibraryHelperFunctions.set_to_value
    where = np.where
    logical_and = np.logical_and
    logical_or  = np.logical_or



class TensorFlowLibrary(ComputationLibrary):
    lib = 'TF'
    reshape = tf.reshape
    permute = tf.transpose
    newaxis = tf.newaxis
    shape = lambda x: x.get_shape()  # .as_list()
    to_numpy = lambda x: x.numpy()
    to_variable = lambda x, dtype: tf.Variable(x, dtype=dtype)
    to_tensor = lambda x, dtype: tf.convert_to_tensor(x, dtype=dtype)
    constant = lambda x, t: tf.constant(x, dtype=t)
    unstack = lambda x, num, axis: tf.unstack(x, num=num, axis=axis)
    ndim = tf.rank
    clip = tf.clip_by_value
    sin = tf.sin
    asin = tf.asin
    cos = tf.cos
    tan = tf.tan
    tanh = tf.tanh
    exp = tf.exp
    squeeze = tf.squeeze
    unsqueeze = tf.expand_dims
    stack = tf.stack
    cast = lambda x, t: tf.cast(x, dtype=t)
    floormod = tf.math.floormod
    floor = tf.math.floor
    ceil = tf.math.ceil
    float32 = tf.float32
    int32 = tf.int32
    int64 = tf.int64
    bool = tf.bool
    tile = tf.tile
    repeat = lambda x, k, a: tf.repeat(x, repeats=k, axis=a)
    gather = lambda x, i, a: tf.gather(x, i, axis=a)
    gather_last = lambda x, i: tf.gather(x, i, axis=-1)
    arange = tf.range
    zeros = tf.zeros
    zeros_like = tf.zeros_like
    ones = tf.ones
    ones_like = tf.ones_like
    sign = tf.sign
    create_rng = lambda seed: tf.random.Generator.from_seed(seed)
    standard_normal = lambda generator, shape: generator.normal(shape)
    uniform = lambda generator, shape, low, high, dtype: generator.uniform(
        shape, minval=low, maxval=high, dtype=dtype
    )
    sum = lambda x, a: tf.reduce_sum(x, axis=a, keepdims=False)
    mean = lambda x, a: tf.reduce_mean(x, axis=a, keepdims=False)
    cumsum = lambda x, a: tf.math.cumsum(x, axis=a)
    cumprod = lambda x, a: tf.math.cumprod(x, axis=a)
    set_shape = lambda x, shape: x.set_shape(shape)
    concat = lambda x, axis: tf.concat(x, axis)
    pi = tf.convert_to_tensor(np.array(np.pi), dtype=tf.float32)
    any = tf.reduce_any
    all = tf.reduce_all
    reduce_any = lambda a, axis: tf.reduce_any(a, axis=axis)
    reduce_all = lambda a, axis: tf.reduce_all(a, axis=axis)
    reduce_max = lambda a, axis: tf.reduce_max(a, axis=axis)
    reduce_min = lambda a, axis: tf.reduce_min(a, axis=axis)
    less = lambda x, y: tf.math.less(x, y)
    less_equal = lambda x, y: tf.math.less_equal(x, y)
    greater = lambda x, y: tf.math.greater(x, y)
    greater_equal = lambda x, y: tf.math.greater_equal(x, y)
    logical_not = lambda x: tf.math.logical_not(x)
    min = tf.minimum
    max = tf.maximum
    atan2 = tf.atan2
    abs = tf.abs
    sqrt = tf.sqrt
    argpartition = lambda x, k: tf.math.top_k(-x, k, sorted=False)[1]
    norm = lambda x, axis: tf.norm(x, axis=axis)
    matmul = tf.linalg.matmul
    cross = tf.linalg.cross
    dot = lambda a, b: tf.tensordot(a, b, 1)
    stop_gradient = tf.stop_gradient
    assign = LibraryHelperFunctions.set_to_variable
    where = tf.where
    logical_and = tf.math.logical_and
    logical_or  = tf.math.logical_or

class PyTorchLibrary(ComputationLibrary):

    @staticmethod
    def gather_last_pytorch(a, index_vector):
        return a[..., index_vector]

    lib = 'Pytorch'
    reshape = torch.reshape
    permute = torch.permute
    newaxis = None
    shape = lambda x: list(x.size())
    to_numpy = lambda x: x.cpu().detach().numpy()
    to_variable = lambda x, dtype: torch.as_tensor(x, dtype=dtype)
    to_tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype)
    constant = lambda x, t: torch.as_tensor(x, dtype=t)
    unstack = lambda x, num, dim: torch.unbind(x, dim=dim)
    ndim = lambda x: x.ndim
    clip = torch.clamp
    sin = torch.sin
    asin = torch.asin
    cos = torch.cos
    tan = torch.tan
    tanh = torch.tanh
    exp = torch.exp
    squeeze = torch.squeeze
    unsqueeze = torch.unsqueeze
    stack = torch.stack
    cast = lambda x, t: x.type(t)
    floormod = torch.remainder
    floor = lambda x: torch.floor(torch.as_tensor(x))
    ceil = lambda x: torch.ceil(torch.as_tensor(x))
    float32 = torch.float32
    int32 = torch.int32
    int64 = torch.int64
    bool = torch.bool
    tile = torch.tile
    repeat = lambda x, k, a: torch.repeat_interleave(x, repeats=k, dim=a)
    gather = lambda x, i, a: torch.gather(x, dim=a, index=i)  # FIXME: It works very differently to TF!!!
    gather_last = gather_last_pytorch
    arange = torch.arange
    zeros = torch.zeros
    zeros_like = torch.zeros_like
    ones = torch.ones
    ones_like = torch.ones_like
    sign = torch.sign
    create_rng = lambda seed: torch.Generator().manual_seed(seed)
    standard_normal = lambda generator, shape: torch.normal(
        torch.zeros(shape), 1.0, generator=generator
    )
    uniform = (
        lambda generator, shape, low, high, dtype: (high - low)
        * torch.rand(*shape, generator=generator, dtype=dtype)
        + low
    )
    sum = lambda x, a: torch.sum(x, a, keepdim=False)
    mean = lambda x, a: torch.mean(x, a, keepdim=False)
    cumsum = lambda x, a: torch.cumsum(x, dim=a)
    cumprod = lambda x, a: torch.cumprod(x, dim=a)
    set_shape = lambda x, shape: x
    concat = lambda x, axis: torch.concat(x, dim=axis)
    pi = torch.from_numpy(np.array(np.pi)).float()
    any = torch.any
    all = torch.all
    reduce_any = lambda a, axis: torch.any(a, dim=axis)
    reduce_all = lambda a, axis: torch.all(a, dim=axis)
    reduce_max = lambda a, axis: torch.max(a, dim=axis)
    reduce_min = lambda a, axis: torch.min(a, dim=axis)[0]
    less = lambda x, y: torch.less(x, y)
    less_equal = lambda x, y: torch.less_equal(x, y)
    greater = lambda x, y: torch.greater(x, y)
    greater_equal = lambda x, y: torch.greater_equal(x, y)
    logical_not = lambda x: torch.logical_not(x)
    min = torch.minimum
    max = torch.maximum
    atan2 = torch.atan2
    abs = torch.abs
    sqrt = torch.sqrt
    argpartition = torch.topk
    norm = lambda x, axis: torch.linalg.norm(x, dim=axis)
    matmul = torch.matmul
    cross = torch.linalg.cross
    dot = torch.dot
    stop_gradient = tf.stop_gradient # FIXME: How to imlement this in torch?
    assign = LibraryHelperFunctions.set_to_value
    where = torch.where
    logical_and = torch.logical_and
    logical_or  = torch.logical_or
