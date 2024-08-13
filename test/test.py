import numpy as np
import sys
sys.path.append('../')
from src.tensor import Tensor

def main():
    ts = [
        Tensor(np.random.uniform(-1., 1., size=(1,5))),
        Tensor(np.random.uniform(-1., 1., size=(5, 1))),
    ]
    print(ts[0].data)
    print(ts[1].data)
    ts.append(ts[0].dot(ts[1]))
    print(ts[2].data)
    ts[2].backward()
    print('--- backward ---')
    print(ts[0].grad, ts[1].grad, ts[2].grad)
    
    print()
    for t in ts:
        if t._ctx is not None:
            for p in t._ctx.parents:
                print('p', p.data)

if __name__ == '__main__':
    main()
