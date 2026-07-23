import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import math

def charfile(file):
    with open(file, 'r') as f:
        words = f.read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    
    return words, chars, stoi, itos

def build_dataset(words, stoi, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

def spliting_dataset(words, stoi, block_size):                                  # training split 80%, dev/validation split 10%, test split 10%
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
    Xte, Yte = build_dataset(words[n2:], stoi, block_size)

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte

# --- classes of tools --
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])   # stacking of parameters
    
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        # parameters (trained with BP)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
        # buffers (updated with a moving average)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)

        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn(num_embeddings, embedding_dim)
    
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    
    def parameters(self):
        return [self.weight]

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    
    def __call__(self, x):
        B, T, C = x.shape           # batch size, blocksize, embedding dimension
        x = x.view(B, T//self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
def network(vocab_size, block_size, n_embd, n_hidden):        # define your model structure
    if math.log2(block_size) != int(math.log2(block_size)):
        raise ValueError("block_size must be a power of 2 for this architecture.")

    n_layer_stacking = int(math.log2(block_size) - 1)
    M = [Embedding(vocab_size, n_embd), FlattenConsecutive(2), Linear(2 * n_embd, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh()]
    for _ in range(n_layer_stacking):
        M += [FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh()]
    M += [Linear(n_hidden, vocab_size)]
    M = Sequential(M)

    # parameter initialization
    with torch.no_grad():
        M.layers[-1].weight *= 0.01
    
    parameters = M.parameters()
    print(f"Number of parameters: {sum(p.numel() for p in parameters)}")
    for p in parameters:
        p.requires_grad = True

    return M, parameters

def main():
    file = '../names.txt'
    words, chars, stoi, itos = charfile(file)
    block_size = 8
    n_embd = 10
    n_hidden = 200
    vocab_size = len(chars) + 1

    Xtr, Ytr, Xdev, Ydev, Xte, Yte = spliting_dataset(words, stoi, block_size)

    # model of neural network
    model, parameters = network(vocab_size, block_size, n_embd, n_hidden)

    # training loop
    max_step = 200001      # max step + 1
    step_track = int(f"{(max_step - 1):e}".split('e')[1]) - 1
    batch_size = 64
    lossi = []

    for i in range(1, max_step):
        # minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward pass
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update: simple SGD
        lr = 0.1 if i < 150000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad
        
        # track stats
        if i == 1 or i % (10 ** step_track) == 0:
            print(f'{i:7d}/{max_step - 1}: {loss.item():.4f}')
        lossi.append((loss.log().item()))

        #break

    # loss curve
    y_axis = torch.tensor(lossi).view(-1, 10 ** ((step_track + 1) // 2 + 1)).mean(1)
    x_axis = list(range(1, len(y_axis) + 1))
    plt.plot(x_axis, y_axis)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss.png')
    plt.close()

    for layer in model.layers:
        layer.training = False
    
    # check the loss on the train, dev and test set
    @torch.no_grad()
    def split_loss(split):
        x, y ={
            'train': (Xtr, Ytr),
            'dev': (Xdev, Ydev),
            'test': (Xte, Yte)
        }[split]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        print(f'{split} loss: {loss.item():.4f}')
    
    split_loss('train')
    split_loss('dev')
    split_loss('test')

    # name generation
    open('generated_names.txt', 'w').close()
    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)        # No need take log, so softmax is enough
            ix = torch.multinomial(probs, num_samples=1).item()
            out.append(ix)
            context = context[1:] + [ix]
            if ix == 0:
                break
        
        #print(''.join(itos[i] for i in out))
        with open('generated_names.txt', 'a') as f:
            f.write(''.join(itos[i] for i in out) + '\n')

if __name__ == '__main__':    
    main()