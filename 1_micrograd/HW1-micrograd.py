# 6-layer MLP with 16 neurons [micrograd]
import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import random
import os

# Define the Value class
class Value:
    def __init__(self, data, _children=(), op='', label=''):
        if isinstance(data, Value):
            data = data.data
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self.op = op
        self.label = label
        self._prev = set(_children)

    def __repr__(self):
        return f'Value({self.data})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, _children=(self,), op=f'**{other}')

        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), op='tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        
# Visualizing the layers
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

def draw_dot(root):
    dat = Digraph(format='svg', graph_attr={'rankdir': 'LR'})   # LR = left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dat.node(name=uid, label='{%s | data %.4f | grad %.4f}' % (n.label, n.data, n.grad), shape='record')
        if n.op:
            dat.node(name=uid + n.op, label=n.op)
            dat.edge(uid + n.op, uid)
    
    for n1, n2 in edges:
        dat.edge(str(id(n1)), str((id(n2))) + n2.op)
    return dat

# Class for nueral network layers
# Class for a single neuron
class Neuron:
    def __init__(self, nin):                                            # nin means the no. of inputs.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]     # \sum_i w_i * x_i + b, sum all inputs x with weights w, and add bias b.
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()                                                # activation function, tanh is used.
        return out
    
    def parameters(self):                                               # return the parameters of the neuron, which are the weights and bias.
        return self.w + [self.b]

# Class for a layer of neurons
class Layer:
    def __init__(self, nin, nout):                                      # nout means the no. of neurons in the layer
        self.nerons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.nerons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.nerons for p in neuron.parameters()]
    
# Class for a multi-layer perceptron
class MLP:
    def __init__(self, nin, nouts):        # nouts is a list of the no. of neurons in each layer, e.g. [4,4,1] means 3 layers with 4, 4, and 1 neurons respectively.
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
# Main program
def main():
# Physics situation: In catalysis, we could input the structures of the catalyst with the adsorbates, pure catalyst, bare adsorbates to obtain each of the energy. To get the interaction enrgy, we can substract their energy: E(abs*) - E(abs) - (E*). In this situation, the input values are the (1) position of the atoms, (2) types of the atoms, (3) cell parameters, etc. Therefore, we could know that the input data are not a single number even though there is only one output value (the interaction energy). In this code, suppose we already had the input and output data. What we want to do is to be familiar with how to develop a multi-layer perceptron (MLP) to correlate the input and output data.  

    #input data
    x = [[6, 18, 29, 34, 37, 38], [4, 6, 26, 28, 34, 40], [1, 5, 6, 25, 30, 42], [10, 17, 19, 28, 45, 49], [3, 12, 23, 28, 35, 38], [7, 23, 28, 31, 33, 34], [9, 17, 27, 34, 39, 47], [1, 2, 4, 30, 41, 43], [7, 10, 11, 19, 25, 30]]   # each list represents a data point with six input values and nine data sets.
    x = [[(x[i][j] - 25) / 8 for j in range(len(x[i]))] for i in range(len(x))]
    
    # output data
    y = [150.5, 130.5, 168.5, 151, 160.5, 196, 127.5, 124.5, 82]        # in big data, the average value is 150. 
    y = [(y[i] - 125)/ 125 for i in range(len(y))]

    n = MLP(6, [16, 16, 16, 16, 16, 16, 1])

    # x = [[2.0, 3.0, -1.0], 
    #   [3.0, -1.0, 0.5], 
    #   [0.5, 1.0, 1.0],
    #   [1.0, 1.0, -1.0]]
    # y = [1.0, -1.0, -1.0, 1.0] # desired targets
    # n = MLP(3, [4, 4, 1])

    for k in range(200):
        # Forward pass
        y_pred = [n(xi) for xi in x]
        loss = sum([(yout - yreal)**2 for yout, yreal in zip(y_pred,y)])
        print(f"iteration {k}, loss = {loss.data}")

        # Backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        for p in n.parameters():
            p.data += -0.001 * p.grad

    x = [[x[i][j] * 8 + 25 for j in range(len(x[i]))] for i in range(len(x))]
    print(x)
    print([y_out.data * 125 + 125 for y_out in y_pred])
    # dot = draw_dot(loss)
    # outpath = os.path.join(os.path.dirname(__file__), "loss_graph")
    # dot.render(outpath, format="png", cleanup=True)

    # Is the model good for prediction? Berify explain your answer in README.md.
    x_new = [2, 10, 13, 16, 20, 21]
    x_new = [(x_new[i] - 25) / 8 for i in range(len(x_new))]
    y_new = n(x_new)
    print(f"Prediction for new data: {y_new.data * 125 + 125}")

if __name__ == '__main__':
    main()