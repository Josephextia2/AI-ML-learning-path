import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import random

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

class MLP(nn.Module):
    def __init__(self, nin, nouts):                 # nin = dimension * block size
        super().__init__()

        sz = [nin] + nouts
        layers = []
        g = torch.Generator().manual_seed(2147483647)
        feature = (5 / 3) / (nin ** 0.5)

        for i in range(len(nouts) - 1):                 # final one is output layer, no activation function    
            layers.append(nn.Linear(sz[i], sz[i + 1], bias=False))     # bias is not needed because BatchNorm1D has bias.
            with torch.no_grad():
                layers[-1].weight *= feature
            layers.append(nn.BatchNorm1d(sz[i + 1], eps=1e-05, momentum=0.01, affine=True, track_running_stats=True, device=None, dtype=None))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(sz[-2], sz[-1]))
        with torch.no_grad():
            layers[-1].weight *= 0.01
            layers[-1].bias.zero_()
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def main():
    file = '../names.txt'
    words, chars, stoi, itos = charfile(file)

    block_size = 3                                                              # 3 words to predict the next word
    n_embd = 10
    n_hidden = 200                                                              # no. of neurons in hidden layer
    layers_no = 1                                                               # no. of hidden layers  
    
    vocab_size = len(chars) + 1

    Xtr, Ytr, Xdev, Ydev, Xte, Yte = spliting_dataset(words, stoi, block_size)

    g = torch.Generator().manual_seed(2147483647)       # for reproducibility
    C = nn.Parameter(torch.randn((vocab_size, n_embd), generator=g))  # characters assigned as arbitrary vectors first: each character -> vector

    network = MLP(block_size * n_embd, [n_hidden] * layers_no + [vocab_size])
    optimizer = torch.optim.Adam(list(network.parameters()) + [C], lr=0.01)

    # training loop
    max_step = 400001       # max step + 1
    step_track = int(f"{(max_step - 1):e}".split('e')[1]) - 1
    batch_size = 64
    lossi = []
    for i in range(max_step):
        # minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,)) 
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward pass
        emb = C[Xb]
        embcat = emb.view(-1, block_size * n_embd)
        logits = network(embcat)
        loss = F.cross_entropy(logits, Yb)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update after backward
        # lr = 0.1 if i < 200000 else 0.01
        # for pg in optimizer.param_groups:
        #     pg['lr'] = lr
        optimizer.step()

        # track stats
        if i % (10 ** step_track) == 0:
            print(f'{i:7d}/{max_step - 1}: {loss.item():.4f}')
        lossi.append((i, loss.log().item()))

        #break # for quick test, remove later

    # plot step vs loss curve
    plt.plot([x[0] for x in lossi], [x[1] for x in lossi])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss.png')
    plt.close()

    # check saturation of hidden layer and dead neurons (1st step)
    # hpreact = network.net[:-3](embcat)
    # h = network.net[-2](hpreact)
    # plt.figure(figsize=(20, 10))
    # plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest') 
    # plt.savefig('test_saturation_neurons.png')
    # plt.close()

    # # plt.figure(figsize=(10, 5))
    # plt.hist(hpreact.view(-1).tolist(), 50);
    # plt.savefig('test_hidden_layer_saturation.png')
    # plt.close()

# check the loss on the train, dev and test set
    network.eval()       # set the model to evaluation mode
    @torch.no_grad()
    def split_loss(split):
        x, y ={
            'train': (Xtr, Ytr),
            'dev': (Xdev, Ydev),
            'test': (Xte, Yte)
        }[split]
        emb = C[x]
        embcat = emb.view(emb.shape[0], -1)
        logits = network(embcat)
        loss = F.cross_entropy(logits, y)
        print(f'{split} loss: {loss.item():.4f}')
    
    split_loss('train')
    split_loss('dev')
    split_loss('test')

    # sample from the model
    g = torch.Generator().manual_seed(2147483647 + 10)

    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            logits = network(embcat)
            probs = F.softmax(logits, dim=1)        # No need take log, so softmax is enough
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            out.append(ix)
            context = context[1:] + [ix]
            if ix == 0:
                break
        
        print(''.join(itos[i] for i in out))
        with open('generated_names_5layers.txt', 'a') as f:
            f.write(''.join(itos[i] for i in out) + '\n')

if __name__ == '__main__':    
    main()