from constants import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
from tqdm import tqdm
import string
import unidecode
import time
import math
import random
import pickle
import tqdm
import traceback
import pandas as pd
import glob
from sklearn.preprocessing import QuantileTransformer
import numpy as np

all_characters = string.printable
n_characters = len(all_characters)
min_len = 260


def read_files():
    files = glob.glob('{0}/{1}/{2}.csv'.format(path, 'text_dumps', '*'))
    dfs = []
    print(files)
    for f in files[:1]:
        try:
            df = pd.read_csv(f, sep = '|')
            dfs.append(df)
        except:
            pass
    df = pd.concat(dfs)

    t = QuantileTransformer()
    df = df[pd.to_numeric(df['score'], errors='coerce').notnull()]
    df['score'] = t.fit_transform(np.reshape(df['score'].values, (-1, 1)))

    df['score'] = df['score'].apply(lambda x: 1.0 if x > 0 else 0.0)
    texts = df.to_dict(orient = 'records')
    print(len(texts))
    print(len(texts))
    return texts

files = read_files()

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def target_tensor(num):
    tensor = torch.from_numpy(np.array([num]))
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=True):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted[-(predict_len * 1):]



class CharRNN_Regressor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN_Regressor, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.fc = nn.Linear(hidden_size*2, 1)


    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden


    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden


    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))



def random_training_set(decoder, criterion, decoder_optimizer, batch_size, chunk_len):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.FloatTensor(batch_size, 1)
    for bi in range(batch_size):
        while True:
            try:

                file = random.choice(files)
                file_t = file['text']

                while len(file_t) + 2 < chunk_len:
                    file = random.choice(files)
                    file_t = file['text']

                chunk = file_t[-chunk_len-1:]
                inp[bi] = char_tensor(chunk[:-1])
                target[bi] = target_tensor(file['score'])
                break
            except:
                # traceback.print_exc()
                pass
    inp = Variable(inp)

    target = Variable(target)
    inp = inp.cuda()
    target = target.cuda()
    return inp, target, decoder, criterion, decoder_optimizer, batch_size, chunk_len


def train(inp, target, decoder, criterion, decoder_optimizer, batch_size, chunk_len):
    hidden = decoder.init_hidden(batch_size)
    hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[:,c], hidden)

    loss += criterion(output, target)
    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / chunk_len

def save(decoder):
    save_filename = os.path.splitext(os.path.basename('torch_char_regressor'))[0]
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


def main():
    epochs = 1000000
    chunk_len = 51
    batch_size = 16
    hidden_size = 256
    n_layers = 5
    print_every = 100

    decoder = CharRNN_Regressor(
        n_characters,
        hidden_size,
        1,
        model='gru',
        n_layers=n_layers,
    )
    decoder_optimizer = torch.optim.RMSprop(decoder.parameters(), lr = 1e-3)
    criterion = nn.MSELoss()

    decoder.cuda()

    start = time.time()
    loss_count = 50

    try:
        print("Training for %d epochs..." % epochs)
        losses = []
        for epoch in range(1, epochs + 1):
            loss = train(*random_training_set(decoder, criterion, decoder_optimizer, batch_size, chunk_len))
            losses.append(loss.item())
            losses = losses[-loss_count:]

            print('epoch:', epoch, 'loss:', sum(losses)/len(losses), len(losses))

            if epoch % print_every == 0:

                print("Saving...")
                save(decoder)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save(decoder)


if __name__ == '__main__':
    main()