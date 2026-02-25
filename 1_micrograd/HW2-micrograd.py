# 6-layer MLP with 16 neurons [pytorch]
import math
import numpy as np
import math
import torch
import torch.nn as nn
from graphviz import Digraph
import random
import os

class MLP(nn.Module):
    def __init__(self, nin, nouts):
        super().__init__()
        sz = [nin] + nouts
        layers = []
        for i in range(len(nouts)):
            layers.append(nn.Linear(sz[i], sz[i+1]))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def main():
    #input data
    x = [[6, 18, 29, 34, 37, 38], [4, 6, 26, 28, 34, 40], [1, 5, 6, 25, 30, 42], [10, 17, 19, 28, 45, 49], [3, 12, 23, 28, 35, 38], [7, 23, 28, 31, 33, 34], [9, 17, 27, 34, 39, 47], [1, 2, 4, 30, 41, 43], [7, 10, 11, 19, 25, 30]]   # each list represents a data point with six input values and nine data sets.
    x = [[(x[i][j] - 25) / 8 for j in range(len(x[i]))] for i in range(len(x))]
    x = torch.tensor(x, dtype=torch.float32) 

    # output data
    y = [150.5, 130.5, 168.5, 151, 160.5, 196, 127.5, 124.5, 82]        # in big data, the average value is 150. 
    y = [(y[i] - 125)/ 125 for i in range(len(y))]
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    n = MLP(6, [16, 16, 16, 16, 16, 16, 1])

    optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
    #criterion = torch.nn.MSELoss()

    for k in range(300):
        optimizer.zero_grad()
        y_pred = n(x)

        #loss = criterion(y_pred, y)
        loss = torch.sum((y_pred - y)**2)
        loss.backward()
        optimizer.step()
        print(f"iteration {k}, loss = {loss.item()}")

    x_input = [x[i][j].item() * 8 + 25 for i in range(len(x)) for j in range(len(x[i]))]
    print(x_input)
    print([y_out.item() * 125 + 125 for y_out in y_pred])
    
    # Is the model good for prediction? Berify explain your answer in README.md.
    x_new = [2, 10, 13, 16, 20, 21]
    x_new = [(x_new[i] - 25) / 8 for i in range(len(x_new))]
    x_new = torch.tensor(x_new, dtype=torch.float32)
    y_new = n(x_new)
    print(f"Prediction for new data: {y_new.item() * 125 + 125}")

if __name__ == '__main__':
    main()