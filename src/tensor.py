# inspired by: https://github.com/tinygrad/tinygrad/blob/c900b6e/tinygrad/tensor.py
import numpy as np
from functools import partialmethod
from typing import Union, List

# TODO: refactor to be more clean (tensor_v2.py)
# TODO: better typing
# TODO: implement lazy eval
class Tensor:
    def __init__(self, data: Union[np.ndarray, List, float]):
        if isinstance(data, np.ndarray): self.data = data.astype(np.float32)
        else: raise TypeError('array has to be of type np.ndarray.')

        self.grad = None
        self._ctx: Context = None

    def backward(self, allow_fill=True):
        if self._ctx is None: return 
        if self.grad is None and allow_fill: 
            assert self.data.size == 1, 'can only start backprop on scalar.'
            self.grad = np.ones_like(self.data)

        assert self.grad is not None

        grads = self._ctx.arg.backward(self._ctx, self.grad)
        grads = [grads] if len(self._ctx.parents) == 1 else grads

        for t, g in zip(self._ctx.parents, grads):
            assert g.shape == t.data.shape, 'grad shape does not match tensor shape.'
            t.grad = g 
            t.backward(False) 
        
class Context:
    def __init__(self, arg_func, *tensors: Tensor):
        self.arg = arg_func
        self.parents = tensors 
        self.saved_tensors = [] 

    def save_for_backward(self, *x: np.ndarray): self.saved_tensors.extend(x)

class Function:
    def forward(ctx: Context, *args): raise NotImplementedError('forward not implemented for function.')
    def backward(ctx: Context, dout: np.ndarray): raise NotImplementedError('backward not implemented for function.')

    def apply(self, arg, *x):
        ctx = Context(arg, self, *x)
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx 
        return ret

def reg_func(name: str):
    def decorator(cls: Function):
        setattr(Tensor, name, partialmethod(cls.apply, cls))
        return cls
    return decorator