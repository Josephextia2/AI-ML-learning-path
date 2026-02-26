Knowledge from seminar: Backward propagation by micrograd. Understanding the correlation between the inputs and outputs in AI language. Each neuron obeys \sum w_i * x_i + b_i, where w and b are weights and bases respectively, while x is input. The weights and bases are adjusted via adding their gradients times learning rate in each iteration. This is imilar to Euler method y_{n+1} = y_{n} + dy_n/dx * dx, dx ~ learning rate. 

Materials: ipynb is the notes of "Neural Networks: Zero to Hero Lesson 1", The spelled-out intro to neural networks and backpropagation: building micrograd.

Homework:
1. (python script, .py file): build a 6-layer MLP with 16 neurons in each layer, and train it on the above dataset. You should be able to get zero loss (or very close to zero loss) after training. [micrograd]

2. same as Homework 1, but using PyTorch. [PyTorch]

References:
1. Neural Networks: Zero to Hero, The spelled-out intro to neural networks and backpropagation: building micrograd; [https://www.youtube.com/watch?v=VMj-3S1tku0]
2. Pytorch official tutorials [https://docs.pytorch.org/tutorials/index.html]
3. AI used for learning: Copilot
