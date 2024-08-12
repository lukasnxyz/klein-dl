# inspired by: https://github.com/tinygrad/tinygrad/blob/c900b6e/tinygrad/tensor.py
import numpy as np
from functools import partialmethod
from typing import Union, List

# TODO: implement lazy eval
class Tensor:
    def __init__(self, data: Union[np.ndarray, List, float], dtype: np.dtype=np.float32):
        # TODO: clean this up
        if type(data) == np.ndarray:
            self.data = data.astype(dtype)
        else: 
            # TODO: raise a TypeError here instead
            print('[WARNING]: make sure all arrays are of type np.ndarray.')
            np.array(data, dtype=dtype)
        self.grad = None
        self._ctx = None
        # TODO: require_grad: bool

    def __str__(self):
        return f'Tensor: {self.data}\ngrad: {self.grad}'

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

class Ctx:
    def __init__(self, arg_func, *tensors: Tensor):
        self.arg = arg_func
        self.parents = tensors 
        self.saved_tensors = [] 

    def save_for_backward(self, *x: np.ndarray):
        self.saved_tensors.extend(x)

class Function:
    def forward(ctx: Ctx, *args): raise NotImplementedError('forward not implemented for function')
    def backward(ctx: Ctx, grad_out: np.ndarray): raise NotImplementedError('backward not implemented for function')

    def apply(self, arg, *x):
        ctx = Ctx(arg, self, *x)
        # call forward of func and create new tensor with result
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx 
        return ret

def register_function(name: str):
    def decorator(cls: type):
        setattr(Tensor, name, partialmethod(cls.apply, cls))
        return cls
    return decorator

# TODO: Add, ReLU, LogSoftmax

@register_function('mul')
class Mul(Function):
    def forward(ctx: Ctx, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x*y

    def backward(ctx: Ctx, dout: np.ndarray):
        x, y = ctx.saved_tensors
        return y*dout, x*dout

@register_function('dot')
class Dot(Function):
    def forward(ctx: Ctx, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x.dot(y)

    def backward(ctx: Ctx, dout: np.ndarray):
        x, y = ctx.saved_tensors 
        dx= dout.dot(y.T) 
        dy= dout.T.dot(x).T 
        return dx, dy
