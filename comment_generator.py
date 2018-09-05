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
import collections
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
# import nltk
# nltk.download('punkt')
all_characters = string.printable
n_characters = len(all_characters)
all_words = []
n_words = 0
g_batch_size = 512
word_dict = dict()
reverse_word_dict = dict()

max_words = 20000
min_len = g_batch_size + 4


def read_files():
    files = glob.glob('{0}/{1}/{2}.csv'.format(path, 'text_dumps', '*'))
    dfs = []
    print(files)
    files = [i for i in files if 'dankmemes' in i]

    for f in files:
        try:
            df = pd.read_csv(f, sep = '|')
            dfs.append(df)
        except:
            pass
    df = pd.concat(dfs)

    t = QuantileTransformer()
    df = df[pd.to_numeric(df['score'], errors='coerce').notnull()]
    df['score'] = t.fit_transform(np.reshape(df['score'].values, (-1, 1)))
    df = df[df['score'] > .5]

    df['text'] = df['text'].fillna('')
    texts = df['text'].tolist()
    # texts = [i for i in texts if len(i) > min_len]
    print(len(texts))
    return texts


def read_sub(sub_name):
    df = pd.read_csv('{0}/{1}/text.csv'.format(path, sub_name), sep='|', engine='python', error_bad_lines=False)
    t = QuantileTransformer()
    df = df[pd.to_numeric(df['score'], errors='coerce').notnull()]
    df['score'] = t.fit_transform(np.reshape(df['score'].values, (-1, 1)))
    df = df[df['score'] > .2]

    df['text'] = df['text'].fillna('')
    texts = df['text'].tolist()
    # texts = [i for i in texts if len(i) > min_len]
    print(len(texts))
    return texts


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
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


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN, self).__init__()
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


def random_training_set(decoder, criterion, decoder_optimizer, batch_size, chunk_len, files):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        while True:
            try:

                file = random.choice(files)
                # while len(file) + 2 < chunk_len:
                #     file = random.choice(files)
                file = ' ' * chunk_len + file
                file_len = len(file)

                start_index = random.randint(0, file_len - chunk_len)
                end_index = start_index + chunk_len + 1
                chunk = file[start_index:end_index]
                inp[bi] = char_tensor(chunk[:-1])
                target[bi] = char_tensor(chunk[1:])
                break
            except:
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
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / chunk_len

def save(decoder, loc = None):
    if not loc:
        save_filename = os.path.splitext(os.path.basename('torch_char_generator'))[0]
    else:
        save_filename = loc + '/torch_char_generator'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)



def make_sub_prediction(sub_name, text, comment_len = 200):
    save_filename = '{0}/{1}/{2}'.format(path, sub_name, 'torch_char_generator')
    decoder = torch.load(save_filename).cpu()
    primer = text
    pred_text = generate(decoder, primer, predict_len = comment_len, temperature=.8, cuda = False)
    return pred_text



def make_sub_model(sub_name):
    files = read_sub(sub_name)

    epochs = 12000
    chunk_len = g_batch_size
    batch_size = 128
    hidden_size = g_batch_size
    n_layers = 5
    print_every = 100

    decoder = CharRNN(
        n_characters,
        hidden_size,
        n_characters,
        model='gru',
        n_layers=n_layers,
    )

    decoder_optimizer = torch.optim.RMSprop(decoder.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    decoder.cuda()
    start = time.time()
    loss_count = 100
    best_score = 10
    patience = 3
    time_since_last_improvement = 0

    print("Training for %d epochs..." % epochs)
    losses = []
    for epoch in range(1, epochs + 1):
        loss = train(*random_training_set(decoder, criterion, decoder_optimizer, batch_size, chunk_len, files))
        losses.append(loss.item())
        losses = losses[-loss_count:]

        print('epoch:', epoch, 'loss:', sum(losses) / len(losses), len(losses))

        if epoch % print_every == 0:
            try:
                file = random.choice(files)
                while len(file) + 2 < chunk_len:
                    file = random.choice(files)
                file_len = len(file)

                start_index = random.randint(0, file_len - chunk_len)
                end_index = start_index + chunk_len + 1
                primer = file[:end_index]
                actual = file[end_index:]

                print()
                print('[%s (%d %d%%) %.4f] %f' % (
                time_since(start), epoch, epoch / epochs * 100, sum(losses) / len(losses), .8))
                pred_text = generate(decoder, primer, 200, temperature=.8)
                print('primer:', primer, '\n')
                print('pred:', pred_text, '\n')
                print('actual:', actual, '\n')

                print('\n\n\n')
            except:
                traceback.print_exc()

            if sum(losses) / len(losses) < best_score:
                best_score = sum(losses) / len(losses)
                time_since_last_improvement = 0

                print("Saving...")
                save(decoder, loc = '{0}/{1}'.format(path, sub_name))
            else:
                time_since_last_improvement += 1

            if patience <= time_since_last_improvement:
                break


def main():
    files = read_files()

    epochs = 25000
    chunk_len = g_batch_size
    batch_size = 128
    hidden_size = g_batch_size
    n_layers = 5
    print_every = 100

    decoder = CharRNN(
        n_characters,
        hidden_size,
        n_characters,
        model='gru',
        n_layers=n_layers,
    )
    decoder_optimizer = torch.optim.RMSprop(decoder.parameters(), lr = 1e-4)
    criterion = nn.CrossEntropyLoss()

    decoder.cuda()

    start = time.time()
    loss_count = 100

    best_score = 10

    try:
        print("Training for %d epochs..." % epochs)
        losses = []
        for epoch in range(1, epochs + 1):
            loss = train(*random_training_set(decoder, criterion, decoder_optimizer, batch_size, chunk_len, files))
            losses.append(loss.item())
            losses = losses[-loss_count:]

            print('epoch:', epoch, 'loss:', sum(losses)/len(losses), len(losses))

            if epoch % print_every == 0:
                try:
                    file = random.choice(files)
                    while len(file) + 2 < chunk_len:
                        file = random.choice(files)
                    file_len = len(file)

                    start_index = random.randint(0, file_len - chunk_len)
                    end_index = start_index + chunk_len + 1
                    primer = file[:end_index]
                    actual = file[end_index:]

                    print()
                    print('[%s (%d %d%%) %.4f] %f' % (time_since(start), epoch, epoch / epochs * 100, sum(losses)/len(losses), .8))
                    pred_text = generate(decoder, primer, 500, temperature=.8)
                    print('primer:', primer, '\n')
                    print('pred:',pred_text, '\n')
                    print('actual:', actual, '\n')

                    print('\n\n\n')
                except:
                    traceback.print_exc()

                if sum(losses)/len(losses) < best_score:
                    best_score = sum(losses)/len(losses)

                    print("Saving...")
                    save(decoder)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == '__main__':
    main()