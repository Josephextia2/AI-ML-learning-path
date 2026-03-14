import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random

def charfile(file):
    with open(file, 'r') as f:
        words = f.read().splitlines()
    
    chars = sorted(list(set(''.join(words))))       # create a to z list
    stoi = {s:i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    return words, chars, stoi, itos

def build_dataset(words, stoi, block_size):      # block size = context length: how many characters do we take to predict the next one?
    X, Y = [], []
    for w in words:                                 # 20 words as an example
        context = [0] * block_size                      # assign [0] * no. of block_size before the start of the words
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]                # shift 1 character to the left
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    return X, Y

def spliting_dataset(words, stoi, block_size):         # training split 80%, dev/validation split 10%, test split 10%
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
    Xte, Yte = build_dataset(words[n2:], stoi, block_size)

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte

def neuron_network(vector_tensor, block_size, chars_no, neurons1_no):        # initialize network parameters; Try MLP later
    g = torch.Generator().manual_seed(2147483647)
    dimension_size = vector_tensor.shape[1]
    w1 = torch.randn((block_size * dimension_size, neurons1_no), generator=g)   # shape = (block size * dimension size, nerons1_no)
    b1 = torch.randn(neurons1_no, generator=g)                                  # shape = (neurons1_no,)
    w2 = torch.randn((neurons1_no, chars_no + 1), generator=g)                  # output layer: 100 neurons -> 27 characters (26 + '.')
    b2 = torch.randn(chars_no + 1, generator=g)
    
    return w1, b1, w2, b2

def main():
    file = 'names.txt'
    block_size = 3
    dimensions = 2                                      # dimension of the embedding vector for each character
    neuron1_no = 400

    words, chars, stoi, itos = charfile(file)
    #X, Y = build_dataset(words, stoi, block_size)      # input and output tensors
    
    
    g = torch.Generator().manual_seed(2147483647)       # for reproducibility
    C = torch.randn(len(chars) + 1, dimensions, generator=g)    # characters assigned as arbitrary vectors first: each character -> vector

    w1, b1, w2, b2 = neuron_network(C, block_size, len(chars), neuron1_no)
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = spliting_dataset(words, stoi, block_size)

    parameters = [C, w1, b1, w2, b2]
    for p in parameters:
        p.requires_grad = True


    # training loop
    iteration = 200000
    step = []
    loss_i = []
    for i in range(iteration):
        # minibatch construction
        ix = torch.randint(0, Xtr.shape[0], (64,))  # 64 samples from Xtr

        # forward pass
        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1, block_size * dimensions) @ w1 + b1) # hidden layer
        logits = h @ w2 + b2                                            # predicted layer/ output layer
        loss = F.cross_entropy(logits, Ytr[ix])

        # backward pass 
        for p in parameters:
            p.grad = None
        loss.backward()

        if i % (iteration / 100) == 0:
            print(f'step {i}: {loss.item()}')

        # update
        lr = 0.1 if i < (iteration / 2) else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        step.append(i)
        loss_i.append(loss.item())
    
    print(f'step {step[-1]}: {loss.item()}')

    # check the loss on the train, dev and test set
    emb_tr, emb_dev, emb_te = C[Xtr], C[Xdev], C[Xte]
    h_tr, h_dev, h_te = torch.tanh(emb_tr.view(-1, block_size * dimensions) @ w1 + b1), torch.tanh(emb_dev.view(-1, block_size * dimensions) @ w1 + b1), torch.tanh(emb_te.view(-1, block_size * dimensions) @ w1 + b1)
    logits_tr, logits_dev, logits_te = h_tr @ w2 + b2, h_dev @ w2 + b2, h_te @ w2 + b2
    loss_tr, loss_dev, loss_te = F.cross_entropy(logits_tr, Ytr), F.cross_entropy(logits_dev, Ydev), F.cross_entropy(logits_te, Yte)
    print(f'loss on train set: {loss_tr.item()}')
    print(f'loss on dev set: {loss_dev.item()}')
    print(f'loss on test set: {loss_te.item()}')


    # plot the step vs loss curve
    plt.plot(step, loss_i)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig("loss.png")
    plt.close()

    # plot character embedding (for spanning in 2 dimensions only)
    plt.figure(figsize=(8,8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha='center', va='center', color='white')
    plt.title('Character Embeddings')
    plt.grid('minor')
    plt.savefig("embedding.png")
    plt.close()

    # sample from the model
    g = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ w1 + b1)
            logits = h @ w2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))



if __name__ == "__main__":
    main()