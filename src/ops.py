import numpy as np
from src.tensor import Function, Context, reg_func

@reg_func('add')
class Add(Function):
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x+y
    
    def backward(ctx: Context, dout: np.ndarray):
        return dout, dout

@reg_func('mul')
class Mul(Function):
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x*y

    def backward(ctx: Context, dout: np.ndarray):
        x, y = ctx.saved_tensors
        return y*dout, x*dout

@reg_func('dot')
class Dot(Function):
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
        ctx.save_for_backward(x, y)
        return x.dot(y)

    def backward(ctx: Context, dout: np.ndarray):
        x, y = ctx.saved_tensors 
        dx= dout.dot(y.T) 
        dy= dout.T.dot(x).T 
        return dx, dy
    
@reg_func('relu')
class ReLU(Function):
    def forward(ctx: Context, x: np.ndarray):
        ctx.save_for_backward(x)
        return np.maximum(x, 0)

    def backward(ctx: Context, dout: np.ndarray):
        x, = ctx.saved_tensors
        dx = dout.copy()
        dx[x<0] = 0
        return dx
    
@reg_func('sum')
class Sum(Function):
    def forward(ctx: Context, x: np.ndarray):
        ctx.save_for_backward(x)
        return np.array([x.sum()])
    
    def backward(ctx: Context, dout: np.ndarray):
        x, = ctx.saved_tensors
        return dout*np.ones_like(x)

@reg_func('logsoftmax')
class LogSoftmax(Function):
    def forward(ctx: Context, x: np.ndarray):
        def logsumexp(x):
            c = x.max(axis=1)
            return c+np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
        out = x-logsumexp(x).reshape((-1, 1))
        ctx.save_for_backward(out)
        return out

    def backward(ctx: Context, dout: np.ndarray):
        out, = ctx.saved_tensors
        return dout-np.exp(out)*dout.sum(axis=1).reshape((-1, 1))