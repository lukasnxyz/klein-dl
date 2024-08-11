import numpy as np
from functools import partialmethod

class Ctx:
    def __init__(self, arg, *tensors):
        self.arg = arg
        self.parents = tensors
        self.saved_tensors = []

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
        if self._ctx is None: return
        if self.grad is None and allow_fill:
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)
        assert(self.grad is not None)
        grads = self._ctx.arg.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print('grad shape must match tensor shape')
                assert(False)
            t.grad = g
            t.backward(False)

class Function:
    def apply(self, arg, *x):
        ctx = Ctx(arg, self, *x)
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx
        return ret

def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

class Dot(Function):
    @staticmethod
    def forward(ctx, inp, weight):
        ctx.save_for_backward(inp, weight)
        return inp.dot(weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T) # shape error sometimes
        grad_weight = grad_output.T.dot(inp).T
        return grad_input, grad_weight
register('dot', Dot)

if __name__ == '__main__':
    t1 = Tensor(np.random.uniform(-1., 1., size=(1,5)))
    print(t1)
    t2 = Tensor(np.random.uniform(-1., 1., size=(5, 1)))
    print(t2)
    t3 = t1.dot(t2)
    print(t3)
    t3.backward()
    print('--- backward ---')
    print(t1)
    print(t2)
    print(t3)
