import numpy as np
from functools import partialmethod
from kleindl.tensor import Operation, Tensor

# TODO: find alternative to subscripting Operation.in_tensors

# assert len(self.in_tensors) == 2
class Dot(Operation):
  def forward(self): 
    return self.in_tensors[0].dot(self.in_tensors[1])

  def backward(self, grad:np.ndarray): 
    return [grad*self.in_tensors[1].data, grad*self.in_tensors[0].data]
Tensor.dot = partialmethod(Tensor._operation_method(Dot))

class Mul(Operation):
  def forward(self): return Tensor()

class ReLU(Operation):
  def forward(self): return Tensor(np.maximum(self.in_tensors[0].data, 0))
  def backward(self, grad:np.ndarray): 
    # compress this to 1 line
    gradc = grad.copy()
    gradc[self.in_tensors[0]<0] = 0
    return gradc
#Tensor.relu = partialmethod(Tensor._operation_method(ReLU))

class LogSoftmax(Operation):
  def forward(self):
    assert len(self.in_tensors) == 1
    x = self.in_tensors[0].data
    # minimize this
    def logsumexp(x):
      c = x.max(axis=1)
      return c+np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    return Tensor(logsumexp(x).reshape((-1, 1)))
    
  def backward(self, grad:np.ndarray):
    out = self.in_tensors[0]
    return grad-np.exp(out)*grad.sum(axis=1).reshape((-1, 1))

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