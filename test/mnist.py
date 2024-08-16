import numpy as np
from tqdm import trange

import sys, os
sys.path.append('../')
from kleindl.tensor import Tensor
from kleindl.ops import *

if os.environ.get('TESTENV') == '1': 
  print('ENV WORKS')
  exit(0)

np.random.seed(42)

mnist_data = dict(np.load('../data/mnist.npz'))
# train
Xtr = mnist_data['x_train'].reshape(60000, -1)
Ytr = mnist_data['y_train']
# val
Xval = mnist_data['x_test'].reshape(10000, -1)
Yval = mnist_data['y_test']

print('shapes', Xtr.shape, Ytr.shape, Xval.shape, Yval.shape)

def mean(self):
  div = Tensor(np.array([1/self.data.size]))
  return self.sum().mul(div)
Tensor.mean = mean

def layer_init(m, h): return np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
class MNIST:
  def __init__(self):
    self.l1 = Tensor(layer_init(784, 128))
    self.l2 = Tensor(layer_init(128, 10))
    
  def forward(self, x):
    return LogSoftmax()(ReLU()(Dot()(Dot()(x, self.l1), self.l2)))
    
model = MNIST()

lr = 0.01
batch_size = 128
lossi, acci = [], []
epochs = 5000

for i in (t := trange(epochs)):
  ix = np.random.randint(0, Xtr.shape[0], size=(batch_size))
  
  x = Tensor(Xtr[ix])
  Y = Ytr[ix]
  y = np.zeros((len(ix), 10), np.float32)
  y[range(y.shape[0]), Y] = -1.0
  y = Tensor(y)
  
  outs = model.forward(x)

  # NLL loss function
  loss = Mul()(outs, y)#.mean()
  loss.backward()
  
  cat = np.argmax(outs.data, axis=1)
  acc = (cat == Y).mean()
  
  # SGD
  model.l1.data = model.l1.data - lr*model.l1.grad
  model.l2.data = model.l2.data - lr*model.l2.grad
  
  # stats
  loss = loss.data
  lossi.append(loss)
  acci.append(acc)
  t.set_description(f'loss {loss.item():.4f} accuracy {acc:.2f}')

# val
def np_eval():
  Y_dev_preds_out = model.forward(Tensor(Xval))
  Y_dev_preds = np.argmax(Y_dev_preds_out.data, axis=1)
  return (Yval == Y_dev_preds).mean()

acc = np_eval()
print(f'val set accuracy is {acc:.2f}')

#import matplotlib.pyplot as plt
#plt.plot(lossi, label='training loss', color='blue')
#plt.plot(acci, label='training accuracy', color='green')
#plt.xlabel('epoch'), plt.legend(), plt.show()