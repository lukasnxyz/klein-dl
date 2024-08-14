import numpy as np
import sys
sys.path.append('../')
from src.tensor import Tensor
from src.ops import *

def main():
    #ts = [
    #    Tensor(np.random.uniform(-1., 1., size=(1,5))),
    #    Tensor(np.random.uniform(-1., 1., size=(5, 1))),
    #    Tensor(np.random.uniform(-1., 1., size=(1, 1)))
    #]
    #ts.append(ts[0].dot(ts[1])) # ts[3]
    #ts.append(ts[2].mul(ts[3])) # ts[4]
    #print('--- backward ---')

    #for p in ts[4]._ctx.parents:
    #    print(p)
    #    #print(p.data)

    #ts[4].backward()
    #for t in ts:
    #    if t._ctx is not None:
    #        for p in t._ctx.parents:
    #            print('p.data', p.data, 'p.grad', p.grad)
    
    #print('---------')
    #for t in ts:
    #    if t._ctx is not None: print(len(t._ctx.saved_tensors))

    n1 = np.random.uniform(-1., 1., size=(1,5))
    n2 = np.random.uniform(-1., 1., size=(5, 1))

    t1 = Tensor(n1, requires_grad=True)
    t2 = Tensor(n2, requires_grad=True)
    t3 = Dot()(t1, t2)
    t3.backward()

    import torch 
    tt1 = torch.Tensor(n1.tolist())
    tt1.requires_grad_()
    tt2 = torch.Tensor(n2.tolist())
    tt2.requires_grad_()
    tt3 = torch.matmul(tt1, tt2)
    tt3.requires_grad_()
    tt3.backward()

    print(tt2.data, tt2.grad)
    print('-------')
    print(t2.data, t2.grad)

if __name__ == '__main__':
    main()
