import numpy as np
import csv
from collections import namedtuple
import json
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from data_index import get_dataset, get_batch
from data_index import info, PERM, TRAIN, VAL, TEST
import os

config = json.load(open("config.json"))
torch.manual_seed(config['seed'])


class Rnn(nn.Module):
    # add self-correcting module later as another nn.Module
    def __init__(self, device="cuda"):
        super(Rnn, self).__init__()
        self.rnn = nn.LSTM(info.feature_size * config["second_split"], hidden_size=config["hidden_size"],
                           num_layers=config["num_layers"], dropout=config["dropout"], batch_first=True)
        self.proj_net = nn.Sequential(
            nn.Linear(config["hidden_size"], config["proj_hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["proj_hidden_size"],
                      info.feature_size * config["second_split"])
        )
        self.to(torch.device(device))

    def forward(self, X, lens):
        # this forward goes through the entire length of the input and spits out all the predictions as output
        # input is NOT a padded sequence
        batch_size, seq_len, ss, fs = X.size()
        X = X.view(batch_size, seq_len, ss*fs)
        packed_X = rnn_utils.pack_padded_sequence(X, lens, batch_first=True)
        out, self.hidden = self.rnn(packed_X)
        out = out.contiguous()
        out = out.view(-1, out.shape[2])
        preds = self.proj_net(out)
        preds = preds.view(config["batch_size"], seq_len, -1)
        # last prediction (can't find it's loss) is still there
        return preds

    def loss(self, X, Y_hat, mask, lens):
        batch_size, seq_len, ss, fs = X.size()
        X = X.view(batch_size, seq_len, ss*fs)
        Y_hat = Y_hat[:, :-1, :]  # 0, 1, ..., last-1
        Y = X[:, 1:, :]  # 1, 2, ..., last
        X = X.view(batch_size, seq_len, ss, fs)
        Y = Y.view(batch_size, seq_len, ss, fs)
        Y_hat = Y_hat.view(batch_size, seq_len, ss, fs)
        loss = (Y - Y_hat) ** 2  # mse loss
        loss = loss * mask  # mask it
        loss = loss.sum() / mask.sum() * (ss * fs)
        return loss


def train_sequence_model(model, optimizer, dataset, data_mask, epochs):

    TOT = (TRAIN // config["batch_size"])
    summary(model, (config["batch_size"], 100, info.feature_size * config["second_split"]))
    for e in range(epochs):
        # loop through batches
        print("Epoch %d" % e)
        for batch_idx in range(TOT):
            batch, mask, lens = get_batch(
                batch_idx, dataset, data_mask, type=0, last=False)
            Y_hat = rnn(batch, lens)
            loss = rnn.loss(batch, Y_hat, mask, lens)

            # output the loss and stuff, all pretty
            print("\t\ttrain loss (%d): %.4f" % (batch_idx, loss.item()))

            # grad step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # final batch (optional)
        batch = get_batch(-1, dataset, data_mask, type=0, last=True)
        Y_hat = rnn(batch, lens)
        loss = rnn.loss(batch, Y_hat, mask, lens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("\t\ttrain loss (last batch): %.4f" % (loss.item()))


def validate(model, dataset, data_mask, final=False):
    loss_sum = 0
    TOT = (VAL // config["batch_size"])
    for batch_idx in range(TOT):
        batch, mask, lens = get_batch(
            batch_idx, dataset, data_mask, type=1, last=False)
        Y_hat = model(batch, lens)
        loss = model.loss(batch, Y_hat, mask, lens)
        loss_sum += loss.item() * config["batch_size"]

    batch = get_batch(-1, dataset, data_mask, type=1, last=True)
    Y_hat = rnn(batch, lens)
    loss = rnn.loss(batch, Y_hat, mask, lens)
    loss_sum += loss.item() * (VAL % config["batch_size"])

    print("Final validation loss")
    print("\tMean:", loss_sum/VAL)


def main():

    load = False
    epochs = 1
    rnn = Rnn()
    optimizer = optim.Adam(rnn.parameters(), lr=config["lr"])
    dataset = torch.load("data/dataset.pt")
    data_mask = torch.load("data/data_mask.pt")
    if load and os.path.exists("checkpoints/" + config["checkpoint_dir"]):
        rnn.load("checkpoints/" + config["checkpoint_dir"])

    train_sequence_model(rnn, optimizer, dataset, data_mask, epochs)
    validate(model, dataset, data_mask, final=True)


if __name__ == "__main__":
    main()

