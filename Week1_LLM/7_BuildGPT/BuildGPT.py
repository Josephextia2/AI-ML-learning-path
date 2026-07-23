import torch
import torch.nn as nn
from torch.nn import functional as F

# sampling (hyperparameters)
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 96
n_head = 6
n_layer = 6
dropout = 0.2

def fileinput(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text

def words(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # mapping from characters to integers
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}

    return chars, vocab_size, stoi, itos



class Head(nn.Module):
    """ self-attention head """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)

    

def main():
    filename = 'input.txt'
    text = fileinput(filename)
    chars, vocab_size, stoi, itos = words(text)

    # encode and decode functions
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # data encode
    data = torch.tensor(encode(text), dtype=torch.long) # 64-bit integer



if __name__ == '__main__':    
    main()