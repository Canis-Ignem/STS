
import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import time
import torch.nn as nn
import math
import numpy as np
from models import TransformerModel, PositionalEncoding
import data_handler as dh


#getting the data

train_data, val_data, test_data, vocab = dh.get_data()

#Hyper params
n_tokens = len(vocab.stoi) # the size of vocabulary
emb_size = 512 # embedding size
n_hidden = 200 # dimension of the FF network inside the transformer
n_layers = 2 # number of transformer layers
n_heads = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # dropout percentage
criterion = nn.CrossEntropyLoss() #loss function
lr = 5.0 # learning rate

# Creating the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = TransformerModel(n_tokens, emb_size, n_heads, n_hidden, n_layers, dropout).to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=lr) #Optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95) #Scheduler for the optimizer




def train(model):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(dh.bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, dh.bptt)):
        data, targets = dh.get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != dh.bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, n_tokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // dh.bptt, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            scheduler.step()


def evaluate(eval_model, data_source):
    #m = np.inner(bench,bench)
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(dh.bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, dh.bptt):
            data, targets = dh.get_batch(data_source, i)
            if data.size(0) != dh.bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            print()
            output_flat = output.view(-1, n_tokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)



####    TRAINING AND VALIDATION

best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):

    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)    #Validating after every training epoch
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    # SAVE THE MODEL THAT PERFORMS BEST IN VALIDATION
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
torch.save(best_model, "transformer_model1.pth")

test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


# Train the model

# The output and the embeddings are 2 different things!

# Can you figure out how to get the embeddings?

# take a look at the shape of the target 
# take a look at the shape of the output
# take a look at the shape of the embedding


# for "yellow"  and "blue" extract the tokens
# put them into a tensor
# get the embeddings of those 2 words
# use np.inner to get the semantic simmilarity 
# what about blue and car?

# can you come up with a different way of computing the semantic simmilarity?



