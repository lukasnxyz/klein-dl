# inspired by: https://github.com/tinygrad/tinygrad/blob/c900b6e/tinygrad/tensor.py
import numpy as np
from functools import partialmethod

class Ctx:
    def __init__(self, arg, *tensors):
        self.arg = arg # Function class child
        self.parents = tensors # len 2, parents of output tensor
        self.saved_tensors = [] # tmp storage 

    # *x is just a tuple of tensors in this case
    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

class Tensor():
    def __init__(self, data):
        # make throw error if not np.ndarray
        self.data = data if type(data) == np.ndarray else np.array(data)
        self.grad = None
        self._ctx = None

    def __str__(self):
        return f'Tensor: {self.data}\ngrad: {self.grad}'

    def backward(self, allow_fill=True):
        # if no context, nothing todo because reached end
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
            if g.shape != t.data.shape:
                print('grad shape must match tensor shape')
                assert(False)
            t.grad = g # gradient of parent tensor
            t.backward(False) # recursivly call backward on parent

class Function:
    def apply(self, arg, *x):
        ctx = Ctx(arg, self, *x)
        # call forward of func and create new tensor with result
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx 
        return ret

def register(name, fxn):
    # add method to Tensor class using partialmethod
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

class Dot(Function):
    @staticmethod
    def forward(ctx, inp, weight):
        ctx.save_for_backward(inp, weight)
        return inp.dot(weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors # get saved tensors
        grad_input = grad_output.dot(weight.T) # grad dot weight
        grad_weight = grad_output.T.dot(inp).T # grad dot inp
        return grad_input, grad_weight
register('dot', Dot) # register Dot function as a method of Tensor

if __name__ == '__main__':
    ts = [
        Tensor(np.random.uniform(-1., 1., size=(1,5))),
        Tensor(np.random.uniform(-1., 1., size=(5, 1))),
    ]
    print(ts[0])
    print(ts[1])
    ts.append(ts[0].dot(ts[1]))
    print(ts[2])
    ts[2].backward()
    print('--- backward ---')
    print(ts[0].grad, ts[1].grad, ts[2].grad)

    print()
    for t in ts:
        if t._ctx is not None:
            for p in t._ctx.parents:
                print(t._ctx.arg)
                print('p', p)
