To understand the inclusion of features to initialize the hidden layers. After passed through a hidden layer, normalization (BatchNorm1D) is considered for avoiding too much dead neurons (saturation of "tanh"). 

Homeworks:

1. Understand the logic of this code

2. Tindy up in notebook in a py script

3. (a) use torch.nn and MLP to simpify the scripts; (b) add to 5 hidden layers to train the model; (c) include examination on the saturation, distributions and learning rate of all layers. Ans: For 5 hidden layers, the loss becomes unstable during training, resulting in invalid training.