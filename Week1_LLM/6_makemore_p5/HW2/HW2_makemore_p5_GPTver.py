import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import math

global vocab_size, block_size, n_embd, n_hidden
vocab_size, block_size, n_embd, n_hidden = None, None, None, None 

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

def no_layer_stacking(block_size, n_hidden, vocab_size):      # exclusion of the input layer
    if math.log2(block_size) != int(math.log2(block_size)):
        raise ValueError("block_size must be a power of 2.")
    
    layer_stacking_array = [n_hidden] * (int(math.log2(block_size) - 1)) + [vocab_size]
    
    return layer_stacking_array


 # --- classes of tools --
class FlattenConsecutive(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch size, blocksize, embedding dimension
        x = x.reshape(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x


class BatchNorm1dLastDim(nn.Module):
    """BatchNorm1d that treats the last dim as channels.

    Matches HW1 behavior:
    - For 2D (B, C): normalize over B.
    - For 3D (B, T, C): normalize over (B, T).
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.bn(x)
        if x.ndim == 3:
            B, T, C = x.shape
            y = self.bn(x.reshape(B * T, C))
            return y.reshape(B, T, C)
        raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")

class MLP(nn.Module):
    def __init__(self, nin, nouts):     # nin is the input (without embedding); nout is no. of neurons array
        super().__init__()

        if vocab_size is None or block_size is None or n_embd is None or n_hidden is None:
            raise ValueError(
                "Global hyperparameters are not initialized. In main(), assign to globals "
                "(vocab_size, block_size, n_embd, n_hidden) before constructing MLP."
            )

        if math.log2(block_size) != int(math.log2(block_size)):
            raise ValueError("block_size must be a power of 2 for this architecture.")

        expected_nin = block_size * n_embd
        if nin != expected_nin:
            raise ValueError(f"nin should be block_size * n_embd = {expected_nin}, got {nin}")

        n_layer_stacking = int(math.log2(block_size) - 1)
        if len(nouts) != n_layer_stacking + 1:
            raise ValueError(
                f"nouts should have length {n_layer_stacking + 1} (hidden repeated + vocab), got {len(nouts)}"
            )
        if nouts[-1] != vocab_size:
            raise ValueError(f"Last element of nouts should be vocab_size={vocab_size}, got {nouts[-1]}")

        # HW1-equivalent stack, but in PyTorch Modules
        layers: list[nn.Module] = [
            nn.Embedding(vocab_size, n_embd),
            FlattenConsecutive(2),
            nn.Linear(2 * n_embd, n_hidden, bias=False),
            BatchNorm1dLastDim(n_hidden),
            nn.Tanh(),
        ]

        for _ in range(n_layer_stacking):
            layers += [
                FlattenConsecutive(2),
                nn.Linear(2 * n_hidden, n_hidden, bias=False),
                BatchNorm1dLastDim(n_hidden),
                nn.Tanh(),
            ]

        layers += [nn.Linear(n_hidden, vocab_size)]

        with torch.no_grad():
            layers[-1].weight *= 0.01
            layers[-1].bias.zero_()

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def main():
    file = '../names.txt'
    words, chars, stoi, itos = charfile(file)
    global vocab_size, block_size, n_embd, n_hidden
    block_size = 8
    n_embd = 10
    n_hidden = 200
    vocab_size = len(chars) + 1

    Xtr, Ytr, Xdev, Ydev, Xte, Yte = spliting_dataset(words, stoi, block_size)

    # model of neural network
    
    nin = block_size * n_embd
    nout = no_layer_stacking(block_size, n_hidden, vocab_size)
    model = MLP(nin, nout)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    

    # training loop
    max_step = 400001      # max step + 1
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
        optimizer.zero_grad()
        loss.backward()

        # update: simple SGD
        optimizer.step()

        # track stats
        if i == 1 or i % (10 ** step_track) == 0:
            print(f'{i:7d}/{max_step - 1}: {loss.item():.4f}')
        lossi.append((loss.log().item()))

        break

    # loss curve
    # y_axis = torch.tensor(lossi).view(-1, 10 ** ((step_track + 1) // 2 + 1)).mean(1)
    # x_axis = list(range(1, len(y_axis) + 1))
    # plt.plot(x_axis, y_axis)
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.savefig('loss.png')
    # plt.close()
    
    # check the loss on the train, dev and test set
    model.eval()       # set the model to evaluation mode
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