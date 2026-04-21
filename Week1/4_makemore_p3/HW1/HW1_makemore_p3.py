import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

def init_neuron_network(vector_tensor, block_size, vocab_size, neurons1_no):         # initialize network parameters; Try MLP later
    g = torch.Generator().manual_seed(2147483647)
    dimension_size = vector_tensor.shape[1]
    feature = (5 / 3) / ((dimension_size * block_size) ** 0.5) 
    
    # Linear parameters
    w1 = torch.randn((block_size * dimension_size, neurons1_no), generator=g) * feature     # shape = (block size * dimension size, nerons1_no)
    #b1 = torch.randn(neurons1_no, generator=g)                                             # shape = (neurons1_no,), wasted parameter
    w2 = torch.randn((neurons1_no, vocab_size), generator=g) * 0.01                         # output layer: 100 neurons -> 27 characters (26 + '.')
    b2 = torch.randn(vocab_size, generator=g) * 0

    # BatchNorm parameters
    bngain = torch.ones((1, neurons1_no))
    bnbias = torch.zeros((1, neurons1_no))
    bnmean_running = torch.zeros((1, neurons1_no))                              # not trainable, but updated
    bnstd_running = torch.ones((1, neurons1_no))

    return w1, w2, b2, bngain, bnbias, bnmean_running, bnstd_running

def main():
    file = '../names.txt'
    words, chars, stoi, itos = charfile(file)

    block_size = 3                                                              # 3 words to predict the next word
    n_embd = 10
    n_hidden = 200
    vocab_size = len(chars) + 1

    g = torch.Generator().manual_seed(2147483647)       # for reproducibility
    C = torch.randn((vocab_size, n_embd), generator=g)  # characters assigned as arbitrary vectors first: each character -> vector

    w1, w2, b2, bngain, bnbias, bnmean_running, bnstd_running = init_neuron_network(C, block_size, vocab_size, n_hidden)
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = spliting_dataset(words, stoi, block_size)

    parameters = [C, w1, w2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True

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
        hpreact = embcat @ w1

        # BatchNorm1D
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
        
        # Tracking_running = True
        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

        h = torch.tanh(hpreact) # hidden layer
        logits = h @ w2 + b2                                            # predicted layer/ output layer
        loss = F.cross_entropy(logits, Yb)

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update after backward
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad  

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
    # plt.figure(figsize=(20, 10))
    # plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest') 
    # plt.savefig('test_saturation_neurons.png')
    # plt.close()

    # plt.figure(figsize=(10, 5))
    # plt.hist(hpreact.view(-1).tolist(), 50);
    # plt.savefig('test_hidden_layer_saturation.png')
    # plt.close()

    # check the loss on the train, dev and test set
    @torch.no_grad()
    def split_loss(split):
        x, y ={
            'train': (Xtr, Ytr),
            'dev': (Xdev, Ydev),
            'test': (Xte, Yte)
        }[split]
        emb = C[x]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ w1
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)
        logits = h @ w2 + b2
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
            hpreact = embcat @ w1
            hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
            h = torch.tanh(hpreact)
            logits = h @ w2 + b2
            probs = F.softmax(logits, dim=1)        # No need take log, so softmax is enough
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            out.append(ix)
            context = context[1:] + [ix]
            if ix == 0:
                break
        
        #print(''.join(itos[i] for i in out))
        with open('generated_names.txt', 'a') as f:
            f.write(''.join(itos[i] for i in out) + '\n')



if __name__ == '__main__':    
    main()