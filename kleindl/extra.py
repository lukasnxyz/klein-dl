from typing import Optional
import time, shutil, sys

# --- tqdm implementation from tinygrad
class tqdm:
  def __init__(self, iterable=None, desc:str='', disable:bool=False, unit:str='it', unit_scale=False, total:Optional[int]=None, rate:int=100):
    self.iter, self.desc, self.dis, self.unit, self.unit_scale, self.rate = iterable, f"{desc}: " if desc else "", disable, unit, unit_scale, rate
    self.st, self.i, self.n, self.skip, self.t = time.perf_counter(), -1, 0, 1, getattr(iterable, "__len__", lambda:0)() if total is None else total
    self.update(0)
  def __iter__(self):
    for item in self.iter:
      yield item
      self.update(1)
    self.update(close=True)
  def set_description(self, desc:str): self.desc = f"{desc}: " if desc else ""
  def update(self, n:int=0, close:bool=False):
    self.n, self.i = self.n+n, self.i+1
    if self.dis or (not close and self.i % self.skip != 0): return
    prog, dur, ncols = self.n/self.t if self.t else 0, time.perf_counter()-self.st, shutil.get_terminal_size().columns
    if self.i/dur > self.rate and self.i: self.skip = max(int(self.i/dur)//self.rate,1)
    def fmt(t): return ':'.join(f'{x:02d}' if i else str(x) for i,x in enumerate([int(t)//3600,int(t)%3600//60,int(t)%60]) if i or x)
    def fn(x): return (f"{x/1000**int(g:=math.log(x,1000)):.{int(3-3*math.fmod(g,1))}f}"[:4].rstrip('.')+' kMGTPEZY'[int(g)].strip()) if x else '0.00'
    unit_text = f'{fn(self.n)}{f"/{fn(self.t)}" if self.t else self.unit}' if self.unit_scale else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
    it_text = (fn(self.n/dur) if self.unit_scale else f"{self.n/dur:5.2f}") if self.n else "?"
    tm = f'{fmt(dur)}<{fmt(dur/prog-dur) if self.n else "?"}' if self.t else fmt(dur)
    suf = f'{unit_text} [{tm}, {it_text}{self.unit}/s]'
    sz = max(ncols-len(self.desc)-5-2-len(suf), 1)
    bar = '\r' + self.desc + (f'{100*prog:3.0f}%|{("█"*int(num:=sz*prog)+" ▏▎▍▌▋▊▉"[int(8*num)%8].strip()).ljust(sz," ")}| ' if self.t else '') + suf
    print(bar[:ncols+1],flush=True,end='\n'*close,file=sys.stderr)

class trange(tqdm):
  def __init__(self, n:int, **kwargs): super().__init__(iterable=range(n), total=n, **kwargs)


# --- graph vis implementation from micrograd
from graphviz import Digraph # brew install graphviz, pip install graphviz
from micrograd.engine import Value

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
# a very simple example
x = Value(1.0)
y = (x * 2 + 1).relu()
y.backward()
draw_dot(y)
# a simple 2D neuron
import random
from micrograd import nn

random.seed(1337)
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
y.backward()

dot = draw_dot(y)
dot
dot.render('gout')

#if __name__ == '__main__':
#  for i in trange(100):
#    time.sleep(0.03)