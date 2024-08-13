import numpy as np
from tqdm import trange

import sys
sys.path.append('../')
from src.tensor import Tensor

np.random.seed(42)

mnist_data = dict(np.load('../data/mnist.npz'))
trlen = mnist_data['y_train'].shape[0]
testlen = mnist_data['y_test'].shape[0]
# split
tds = int(testlen*0.5)
# train
Xtr = mnist_data['x_train'].reshape(60000, -1)
Ytr = mnist_data['y_train'].reshape(-1, 1)
# dev
Xdev = mnist_data['x_test'].reshape(10000, -1)[:, :tds]
Ydev = mnist_data['y_test'].reshape(-1, 1)[:, :tds]
# test
Xtest = mnist_data['x_test'].reshape(10000, -1)[:, tds:]
Ytest = mnist_data['y_test'].reshape(-1, 1)[:, tds:]

print('shapes', Xtr.shape, Ytr.shape, Xdev.shape, Ydev.shape, Xtest.shape, Ytest.shape)

def layer_init(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
    return ret

class MNIST:
    def __init__(self):
        self.l1 = Tensor(layer_init(784, 128))
        self.l2 = Tensor(layer_init(128, 10))
    
    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()
    
model = MNIST()

lr = 0.1
batch_size = 128
lossi, acci = [], []
epochs = 1

for i in (t := trange(epochs)):
  ix = np.random.randint(0, Xtr.shape[0], size=(batch_size))
  
  x = Tensor(Xtr[ix])
  Y = Ytr[ix]
  y = np.zeros((len(ix), 10), np.float32)
  y[range(y.shape[0]), Y] = -1.0
  y = Tensor(y)
  
  # network
  outs = model.forward(x)

  # NLL loss function
  loss = outs.mul(y).mean()
  loss.backward()
  
  cat = np.argmax(outs.data, axis=1)
  print(cat.shape)
  print(cat[:20])
  acc = (cat == Y).mean()
  
  # SGD
  model.l1.data = model.l1.data - lr*model.l1.grad
  model.l2.data = model.l2.data - lr*model.l2.grad
  
  # printing
  loss = loss.data
  lossi.append(loss)
  acci.append(acc)
  t.set_description(f'loss {loss.item():.4f} accuracy {acc:.2f}')

#import matplotlib.pyplot as plt
#plt.plot(lossi)
#plt.show()

## evaluate
#def numpy_eval():
#  Y_dev_preds_out = model.forward(Tensor(Xdev))
#  print(Y_dev_preds_out.data[:20])
#  Y_dev_preds = np.argmax(Y_dev_preds_out.data, axis=1)
#  return (Ydev== Y_dev_preds).mean()
#
#accuracy = numpy_eval()
#print(f'dev set accuracy is {accuracy:.2f}')
#assert accuracy > 0.95