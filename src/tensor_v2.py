# inspired by: https://github.com/tinygrad/tinygrad/blob/c900b6e/tinygrad/tensor.py
import numpy as np
from functools import partialmethod
from typing import Union, List, Callable

GDTYPE = np.float32

class Tensor:
    def __init__(self, data: Union[np.ndarray, List, float]):
        if isinstance(data, np.ndarray): self.data = data.astype(GDTYPE)
        else: raise TypeError('array has to be of type np.ndarray.')
        self.grad: np.ndarray = None
        self._ctx: Context = None

    def backward(self, allow_fill=True):
        pass

class Context:
    def __init__(self, function: Callable, *tensors: Tensor):
        self.function = function
        self.parents = tensors
        self.saved_tensors = []

    def save(self, *x: np.ndarray): self.saved_tensors.extend(x)

class Function:
    def forward(ctx: Context, *arg_tensors: np.ndarray): raise NotImplementedError('forward not implemented for function.')
    def backward(ctx: Context, dout: np.ndarray): raise NotImplementedError('backward not implemented for function.')
    def apply(self, function: Callable, *tensors: Tensor):
        ctx = Context(function, self, *tensors)
        out = Tensor(function.forward(ctx, self.data, *[t.data for t in tensors]))
        out._ctx = ctx 
        return out

def reg_func(name: str):
    def decorator(cls: type):
        setattr(Tensor, name, partialmethod(cls.apply, cls))
        return cls
    return decorator