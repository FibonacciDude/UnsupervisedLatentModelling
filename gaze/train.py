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
import data_index
import math
#import data_index.info as info
import os

info = data_index.info
config = json.load(open("config.json"))
torch.manual_seed(config['seed'])


class Rnn(nn.Module):
    # add self-correcting module later as another nn.Module
    def __init__(self, device="cuda"):
        super(Rnn, self).__init__()
        self.x_size = info.feature_size * config["second_split"]
        # proj size = hidden size
        self.rnn = nn.LSTM(self.x_size, hidden_size=config["hidden_size"],
                           num_layers=config["num_layers"], dropout=config["dropout"], batch_first=True)
        self.proj_net = nn.Sequential(
            nn.ReLU(),
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
        packed_X = rnn_utils.pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        out, self.hidden = self.rnn(packed_X)
        #unpack
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        #project
        out = self.proj_net(out)
        out = out.contiguous()
        out = out.view((batch_size, seq_len, -1))
        # last prediction (can't find it's loss) is still there
        return out

    def loss(self, X, Y_hat, mask):
        batch_size, seq_len, ss, fs = X.size()
        X = X.view(batch_size, seq_len, ss*fs)
        Y_hat = Y_hat[:, :-1, :]  # 0, 1, ..., last-1
        Y = X[:, 1:, :]  # 1, 2, ..., last
        #X = X.view(batch_size, seq_len, ss, fs)
        Y = Y.view(batch_size, seq_len-1, ss, fs)
        Y_hat = Y_hat.view(batch_size, seq_len-1, ss, fs)
        mask = mask[:, :-1, :, :]
        loss = (Y - Y_hat) ** 2  # mse loss
        loss = loss * mask  # mask it
        loss = loss.sum()  / mask.sum() * (ss * fs) * config["loss_factor"]
        return loss

def summary(model, name):
    num=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters in model %s:" % name, num)

def loop(model, optimizer, dataset, data_mask, data_lens, TOT, type_=0, update=True):
    losses = []
    for batch_idx in range(TOT):
        batch, mask, lens = get_batch(
            batch_idx, dataset, data_mask, data_lens, type=type_, last=False)
        Y_hat = model(batch, lens)
        loss = model.loss(batch, Y_hat, mask)

        losses.append(loss.item() * Y_hat.size()[0] / config["batch_size"])
        if batch_idx % config["print_every"] == 0 and type_==0:
            # output the loss and stuff, all pretty
            print("\t\ttrain loss (%d): %.4f" % (batch_idx, loss.item()))

        if update:
            # grad step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return np.array(losses)

def train_sequence_model(model, optimizer, dataset, data_mask, data_lens, epochs):

    TOT = math.ceil(data_index.TRAIN / config["batch_size"])
    #summary(model, (config["batch_size"], 100, info.feature_size * config["second_split"]))
    summary(model, "rnn")
    plot_train = []
    plot_val = []
    for e in range(epochs):
        # loop through batches
        print("Epoch %d" % e)
        losses = loop(model, optimizer, dataset, data_mask, data_lens, TOT=TOT, type_=0, update=True)
        print("Epoch %d over, average loss" % e, losses.mean())
        val_losses = validate(model, dataset, data_mask, data_lens, final=True)
        print("\t\tVal loss: %.4f" % (val_losses.mean()))
        plot_val.append(val_losses.mean()) 
        plot_train.append(losses.mean()) 
    return plot_train, plot_val

def validate(model, dataset, data_mask, data_lens, final=False):
    TOT = (data_index.VAL // config["batch_size"])
    losses = loop(model, None, dataset, data_mask, data_lens, TOT=TOT, type_=1, update=False)

    if final:
        print("Final validation")
        print("\tMean validation loss:", losses.mean())
    return losses


def main():
    load = False
    epochs = 10
    rnn = Rnn()
    optimizer = optim.Adam(rnn.parameters(), lr=config["lr"], weight_decay=config["l2_reg"])
    dataset = torch.load("data/dataset.pt")
    data_mask = torch.load("data/data_mask.pt")
    lens = torch.load("data/lens.pt")
    if load and os.path.exists("checkpoints/" + config["checkpoint_dir"]):
        rnn.load("checkpoints/" + config["checkpoint_dir"])

    plt_t, plt_v = train_sequence_model(rnn, optimizer, dataset, data_mask, lens, epochs)
    print(plt_t)
    print(plt_v)
    validate(rnn, dataset, data_mask, lens, final=True)


if __name__ == "__main__":
    main()
