# inspired by: https://github.com/tinygrad/tinygrad/blob/c900b6e/tinygrad/tensor.py
import numpy as np
from typing import Union, List, Optional

GDTYPE = np.float32

class Tensor:
  def __init__(self, data:Union[np.ndarray, List, float], requires_grad:Optional[bool]=False):
    if isinstance(data, np.ndarray): self.data = data.astype(GDTYPE)
    else: raise TypeError('array has to be of type np.ndarray.')

    #TODO: if !requires_grad: don't have self.grad, self._ctx
    self.requires_grad = requires_grad
    self.grad: np.ndarray = None
    self.parents = []
    self.operation = None
    
  @classmethod
  def _operation_method(cls, operation):
    def method(self, other): return operation()(self, other)
    return method
    
  def backward(self):
    if not self.requires_grad: return
    if self.grad is None: self.grad = np.ones_like(self.data)
    visited = set()
    def traverse(t):
      if t in visited: return
      visited.add(t)
      if t.operation:
        grads = t.operation.backward(t.grad)
        for parent, grad in zip(t.parents, grads):
          if parent.grad is None: parent.grad = grad
          else: parent.grad += grad
          traverse(parent)
    traverse(self)

# TODO:
# basically need to check if the operation is a binary or unary op
# if it's a unary operation, avoid checking NoneType

class Operation:
  def __call__(self, *in_tensors:Optional[Tensor]):
    #print('-->', len(in_tensors))
    self.in_tensors = [t for t in in_tensors]
    self.out = self.forward()

    # -- handle this 
    if None in self.in_tensors:
      self.in_tensors.remove(None)
    # --

    if any(t.requires_grad for t in self.in_tensors):
      self.out.requires_grad = True
      self.out.operation = self
      self.out.parents = self.in_tensors
    return self.out
  def forward(self): raise NotImplementedError
  def backward(self, grad): raise NotImplementedError