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

all_characters = string.printable
n_characters = len(all_characters)
min_len = 150

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


def get_comments(prev_text, comments):
    try:
        prev_text = prev_text + c_splitter + str(comments.body)
        comment_text = []

        for i in comments.replies._comments:
            comment_text.extend(get_comments(prev_text, i))

        return comment_text
    except:
        # traceback.print_exc()
        return [prev_text]


def read_files():
    with open(path + 'posts.plk', 'rb') as f:
        posts = pickle.load(f)
    random.shuffle(posts)

    texts = []
    for i in tqdm.tqdm(posts[:1000]):
        post_title = i.title
        for j in i.comments._comments:
            try:
                texts.extend(get_comments(post_title, j))
            except:
                traceback.print_exc()
    texts = [i for i in texts if len(i) > min_len]
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


def random_training_set(decoder, criterion, decoder_optimizer, batch_size, chunk_len):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        while True:
            try:

                file = random.choice(files)
                while len(file) + 2 < chunk_len:
                    file = random.choice(files)
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

def save(decoder):
    save_filename = os.path.splitext(os.path.basename('torchmodel'))[0]
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


def main():
    epochs = 1000000
    chunk_len = 256
    batch_size = 512
    hidden_size = 128
    n_layers = 5
    print_every = 100

    decoder = CharRNN(
        n_characters,
        hidden_size,
        n_characters,
        model='gru',
        n_layers=n_layers,
    )
    decoder_optimizer = torch.optim.RMSprop(decoder.parameters())
    criterion = nn.CrossEntropyLoss()

    decoder.cuda()

    start = time.time()
    all_losses = []
    loss_avg = 0

    try:
        print("Training for %d epochs..." % epochs)
        for epoch in range(1, epochs + 1):
            print('epoch:', epoch)
            loss = train(*random_training_set(decoder, criterion, decoder_optimizer, batch_size, chunk_len))
            loss_avg += loss

            if epoch % print_every == 0:
                for t in [.8]:
                    file = random.choice(files)
                    while len(file) + 2 < chunk_len:
                        file = random.choice(files)
                    file_len = len(file)

                    start_index = random.randint(0, file_len - chunk_len)
                    end_index = start_index + chunk_len + 1
                    primer = file[start_index:end_index]

                    print()
                    print('[%s (%d %d%%) %.4f] %f' % (time_since(start), epoch, epoch / epochs * 100, loss, t))
                    pred_text = generate(decoder, primer, 500, temperature=t)
                    print('primer:', primer, '\n')
                    print('pred:',pred_text, '\n')
                    print('\n\n\n')

                print("Saving...")
                save(decoder)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save(decoder)


if __name__ == '__main__':
    main()