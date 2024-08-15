import numpy as np
import sys
sys.path.append('../')
import kleindl

def main():
    n1 = np.random.uniform(-1., 1., size=(1,5))
    n2 = np.random.uniform(-1., 1., size=(5, 1))

    t1 = kleindl.Tensor(n1, requires_grad=True)
    t2 = kleindl.Tensor(n2, requires_grad=True)
    t3 = t1.dot(t2)
    t3.relu(None)
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
