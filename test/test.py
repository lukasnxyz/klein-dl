import numpy as np
import sys
sys.path.append('../')
from src.tensor import Tensor

def main():
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
                print('p', p)

if __name__ == '__main__':
    main()
