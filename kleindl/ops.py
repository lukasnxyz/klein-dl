import numpy as np
from functools import partialmethod
from kleindl.tensor import Operation, Tensor

# TODO: better way to convert forward to tensor
# TODO: minimize logsoftmax
# TODO: backward returns Tensor as well
class Dot(Operation):
  def forward(self) -> Tensor:
    return Tensor(self.saved[0].dot(self.saved[1]))

  def backward(self, grad:np.ndarray):
    return [grad*self.saved[1], grad*self.saved[0]]

class ReLU(Operation):
  def forward(self) -> Tensor:
    return Tensor(np.maximum(self.saved[0], 0))

  def backward(self, grad:np.ndarray):
    return np.where(self.saved[0] < 0, 0, grad)

class Mul(Operation):
  def forward(self) -> Tensor:
    return Tensor(self.saved[0]*self.saved[1])
    
  def backward(self, grad:np.ndarray):
    return self.saved[0]*grad, self.saved[1]*grad
Tensor.mul = partialmethod(Tensor._operation_method(Mul))
  
class Sum(Operation):
  def forward(self) -> Tensor:
    return Tensor(np.array([self.saved[0].sum()]))
  
  def backward(self, grad:np.ndarray):
    return grad*np.ones_like(self.saved[0])
Tensor.sum = partialmethod(Tensor._operation_method(Sum))

class LogSoftmax(Operation):
  def forward(self) -> Tensor:
    x = self.saved[0]
    # minimize this
    def logsumexp(x):
      c = x.max(axis=1)
      return c+np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    return Tensor(logsumexp(x).reshape((-1, 1)))
    
  def backward(self, grad:np.ndarray):
    out = self.saved[0]
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