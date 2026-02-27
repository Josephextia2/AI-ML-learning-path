import math
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

name_number = 40                 # no. of names to generate
training_iterations = 200

def main():
    file = "./names.txt"
    with open(file, 'r') as f:
        words = f.read().splitlines()
    
    # assign an integer to each character
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i, s in enumerate(chars)}         #string to integers
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    #print(stoi)

    # add . to beginning and end of each word
    # add x_source and y_source for training data
    N = torch.zeros((27, 27), dtype=torch.int32)        # array for counting the bigrams
    xs, ys = [], []                                     # x_source and y_source 
    for w in words:
        chs = ['.'] + list(w) +['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]                             # for locating the position of element in the array N
            ix2 = stoi[ch2]                             
            N[ix1, ix2] += 1                            # counting
            xs.append(ix1)
            ys.append(ix2)
    
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    #total_number = xs.nelement()                        # total number of bigrams
    #print(f"Total number of bigrams: {total_number}")


    #visualization
    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
            plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
    plt.axis('off')
    plt.savefig('bigrams.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Probability of bigrams; generate names according to probability distribution
    g = torch.Generator().manual_seed(2147483647)           # Create a random generator with a fixed seed for reproducibility
    P = (N + 1).float()
    P /= P.sum(1, keepdim=True)                             # Normalization; noted that adding 1 on both numerator and denominator doesn't really effect the probabilities, but it does prevent zero probabilities.
    name_actual = []     
    for i in range(name_number):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        #print(''.join(out))
        name_actual.append(''.join(out))
    #print(P[0])
    logprob = torch.log(P)
    log_likehood = - (logprob * N).sum() / N.sum()
    print(f"Average negative log likelihood per bigram: {log_likehood}")
    
    

    # Deep learning: generate names according to the a previous character, consistent with the probabilities of bigrams.
    g = torch.Generator().manual_seed(2147483647) 
    xenc = F.one_hot(xs, num_classes=27).float()
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    loss_graph = []
    learning_rate_graph = []
    
    for j in range(training_iterations):
        # forward pass
        logits = xenc @ W                                   # assume as log-counts
        counts = logits.exp()       
        probs = counts / counts.sum(1, keepdim=True)        # probabilities during training; not yet sum the repeated bigrams because shape = (xs.nelement(), 27).

        pred_log_likehood = -probs[torch.arange(len(xs)), ys].log().mean() + 0.01 * (W**2).mean()     # First term can be comparable to log_likehood; L2 regularization
        loss = pred_log_likehood - log_likehood
        #print(loss.item())

        # backward pass
        W.grad = None
        loss.backward()

        # update
        lr = 100 * math.exp(-j/100)    # learning rate decay
        W.data += -lr * W.grad.data

        loss_graph.append(loss.item())
        learning_rate_graph.append(lr)

    # plot the loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(loss_graph)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.savefig("training_loss.png")
    plt.close()

    # plot the learning rate graph
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rate_graph)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Over Time")
    plt.savefig("learning_rate.png")
    plt.close()

    # sample from the model
    g = torch.Generator().manual_seed(2147483647)
    name_predicted = [] 
    for i in range(name_number):
        out = []
        ix = 0
        while True:
            xenc_pred = F.one_hot(torch.tensor([ix]), num_classes=27).float()                # start from "."
            logits_pred = xenc_pred @ W
            count_pred = logits_pred.exp()
            prob_pred = count_pred / count_pred.sum(1, keepdim=True)
            ix = torch.multinomial(prob_pred, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        #print(''.join(out))
        name_predicted.append(''.join(out))

    with open('generated_names.txt', 'w') as f:
        f.write("Actual names:\n")
        for name in name_actual:
            f.write(name + "\n")
        f.write("\n")
        f.write("\nPredicted names:\n")
        for name in name_predicted:
            f.write(name + "\n")

if __name__ == "__main__":
    main()