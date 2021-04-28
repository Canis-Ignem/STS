import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

bptt = 35
batch_size = 20
eval_batch_size = 10
tokenizer = get_tokenizer('basic_english')

def data_process(raw_text_iter, vocab ):
  data = [torch.tensor([vocab[token] for token in tokenizer(item) ], dtype=torch.long ) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))



def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device)

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target




def get_data():

    train_iter = WikiText2(split='train')
    counter = Counter()
    for line in train_iter:
        counter.update(tokenizer(line))
    vocab = Vocab(counter)

    train_iter, val_iter, test_iter = WikiText2()
    '''
    i = 0
    for item in train_iter:
        print(item)
        if i == 5:
            break
    
        i+=1
    '''
    train_data = data_process(train_iter, vocab)
    val_data = data_process(val_iter, vocab)
    test_data = data_process(test_iter, vocab)

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)
    return train_data, val_data, test_data, vocab





