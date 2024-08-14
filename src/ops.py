import numpy as np
#from src.tensor import Operation, Context, reg_func
from src.tensor import Operation, Tensor

class Dot(Operation):
    def forward(self):
        return Tensor(np.dot(self.in_tensors[0].data, self.in_tensors[1].data))
    
    def backward(self, grad):
        return [grad*self.in_tensors[1].data, grad*self.in_tensors[0].data]

#@reg_func('dot')
#class Dot(Operation):
#    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
#        ctx.save(x, y)
#        return x.dot(y)
#
#    def backward(ctx: Context, dout: np.ndarray):
#        x, y = ctx.parents 
#        dx= dout.dot(y.T) 
#        dy= dout.T.dot(x).T 
#        return dx, dy
# 
#@reg_func('add')
#class Add(Operation):
#    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
#        ctx.save(x, y)
#        return x+y
#    
#    def backward(ctx: Context, dout: np.ndarray):
#        return dout, dout
#
#@reg_func('mul')
#class Mul(Operation):
#    def forward(ctx: Context, x: np.ndarray, y: np.ndarray):
#        ctx.save(x, y)
#        return x*y
#
#    def backward(ctx: Context, dout: np.ndarray):
#        print(len(ctx.parents))
#        x, y = ctx.parents
#        return y*dout, x*dout
#
#@reg_func('relu')
#class ReLU(Operation):
#    def forward(ctx: Context, x: np.ndarray):
#        ctx.save(x)
#        return np.maximum(x, 0)
#
#    def backward(ctx: Context, dout: np.ndarray):
#        x, = ctx.parents
#        dx = dout.copy()
#        dx[x<0] = 0
#        return dx
#    
#@reg_func('sum')
#class Sum(Operation):
#    def forward(ctx: Context, x: np.ndarray):
#        ctx.save(x)
#        return np.array([x.sum()])
#    
#    def backward(ctx: Context, dout: np.ndarray):
#        x, = ctx.parents
#        return dout*np.ones_like(x)
#
#@reg_func('logsoftmax')
#class LogSoftmax(Operation):
#    def forward(ctx: Context, x: np.ndarray):
#        def logsumexp(x):
#            c = x.max(axis=1)
#            return c+np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
#        out = x-logsumexp(x).reshape((-1, 1))
#        ctx.save(out)
#        return out
#
#    def backward(ctx: Context, dout: np.ndarray):
#        out, = ctx.parents
#        return dout-np.exp(out)*dout.sum(axis=1).reshape((-1, 1))