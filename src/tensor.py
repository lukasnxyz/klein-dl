# inspired by: https://github.com/tinygrad/tinygrad/blob/c900b6e/tinygrad/tensor.py
import numpy as np
from functools import partialmethod
from typing import Union, List

class Ctx:
    def __init__(self, arg_func, *tensors):
        self.arg = arg_func
        self.parents = tensors 
        self.saved_tensors = [] 

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

# TODO: implement lazy eval
class Tensor:
    def __init__(self, data: Union[np.ndarray, List, float], dtype=np.float32):
        # TODO: clean this up
        if type(data) == np.ndarray:
            self.data = data.astype(dtype)
        else: 
            # TODO: raise a TypeError here instead
            print('[WARNING]: make sure all arrays are of type np.ndarray')
            np.array(data, dtype=dtype)
        self.grad = None
        self._ctx = None

    def __str__(self):
        return f'Tensor: {self.data}\ngrad: {self.grad}'

    def backward(self, allow_fill=True):
        if self._ctx is None: return 
        # only start backprop on a scalar (loss)
        if self.grad is None and allow_fill: 
            assert self.data.size == 1 
            self.grad = np.ones_like(self.data)

        assert(self.grad is not None)

        # compute grads with backward of the function
        grads = self._ctx.arg.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads] # wrap single gradient in a list
        for t, g in zip(self._ctx.parents, grads):
            # TODO: shape check doesn't work
            if g.shape != t.data.shape: 
                print('grad shape does not match tensor shape')
                assert(False)
            t.grad = g # gradient of parent tensor
            t.backward(False) # recursivly call backward on parent

class Function:
    def forward(ctx, *args): raise NotImplementedError('forward not implemented for function')
    def backward(ctx, grad_output): raise NotImplementedError('backward not implemented for function')

    def apply(self, arg, *x):
        ctx = Ctx(arg, self, *x)
        # call forward of func and create new tensor with result
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx 
        return ret

def register_function(name):
    def decorator(cls):
        setattr(Tensor, name, partialmethod(cls.apply, cls))
        return cls
    return decorator

# TODO: Add, ReLU, LogSoftmax

@register_function('mul')
class Mul(Function):
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x*y

    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y*grad_output, x*grad_output

@register_function('dot')
class Dot(Function):
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x.dot(y)

    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors 
        grad_input = grad_output.dot(y.T) 
        grad_weight = grad_output.T.dot(x).T 
        return grad_input, grad_weight
