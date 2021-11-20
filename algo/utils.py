import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.distributions.normal import Normal 
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import logging # implement logger

logger = logging.getLogger(__name__)
device = torch.device("cuda")
dtype = torch.float64

class Config():
    def __init__(self,
            x_dim,
            action_dim,
            N, 
            hidden_size,
            num_layers,
            dir,
            trajectory=1,
            checkpoint_dir="model.pth",
            dropout=0,
            gamma=.99,
            beta=0,
            window_size=100,
            amt_actions=1,
            lr=1e-3,
            beta_decay=.9,
            normalization_window=20,
            continuous=True,
            value_function=False,
            device="cuda",
            dtype=torch.float64,
            phi=lambda x:x):

       # gpu
       self.device = torch.device(device)
       self.dtype = dtype

       # input, output dimensions
       self.x_dim = x_dim 
       self.action_dim = action_dim
       self.continuous = continuous

       # batch_size
       self.N = N
       self.lr = lr
       
       # dataset params
       self.window_size=window_size
       self.normalization_window=normalization_window

       self.value_function = value_function

       self.phi = phi
       self.amt_actions = amt_actions
       
       # gamma, beta
       self.gamma = gamma
       self.beta = beta
       self.beta_decay = beta_decay
       # LSTM Analyzer params
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.dropout = 0 
       
       #housekeeping
       self.dir = dir
       self.checkpoint_dir = checkpoint_dir
       self.trajectory = trajectory

class Dataset():
    def __init__(self, config):
        self.config = config
        dtype, device = config.dtype, config.device
        self.states = torch.rand((0, config.N, config.x_dim), dtype=dtype, device=device, requires_grad=False)
        self.log_probs_shape = (config.amt_actions,) + config.continuous * (config.action_dim,)
        self.log_probs = torch.rand((0, config.N) + self.log_probs_shape, dtype=dtype, device=device, requires_grad=False)

        self.rewards = torch.rand((0, config.N, 1), dtype=dtype, device=device, requires_grad=False)
        self.hidden_h = torch.rand((1, config.num_layers, config.N, config.action_dim*config.amt_actions), dtype=dtype, device=device, requires_grad=False)
        self.hidden_c = torch.rand((1, config.num_layers, config.N, config.hidden_size), dtype=dtype, device=device, requires_grad=False)
        self.reset_storage()

    def reset_storage(self):
        config = self.config
        self.store_log_probs = torch.rand((0, config.N) + self.log_probs_shape, dtype=dtype, device=device, requires_grad=True)
        self.store_mean = torch.rand((0, config.N) + self.log_probs_shape, dtype=dtype, device=device, requires_grad=True)
        self.store_std = torch.rand((0, config.N) + self.log_probs_shape, dtype=dtype, device=device, requires_grad=True)
        self.store_hidden_h = torch.rand((0, config.num_layers, config.N, config.action_dim*config.amt_actions), dtype=dtype, device=device, requires_grad=True)
        self.store_hidden_c = torch.rand((0, config.num_layers, config.N, config.hidden_size), dtype=dtype, device=device, requires_grad=True)
        
    @property
    def mean_std_rewards(self):
        self.rewards.detach_()
        std, mean = torch.std_mean(self.rewards[-self.config.normalization_window:].squeeze(2), 0)
        return mean, std

    @property
    def std(self):
        return (self.store_std)

    @property
    def context_size(self):
        return self.states.size()[0]

    def add(self, z, log_probs, h_c):
        h, c = h_c
        # store, don't place into requires_grad=False tensor until backprop
        self.store_hidden_h = torch.cat((self.store_hidden_h, h), dim=0)
        self.store_hidden_c = torch.cat((self.store_hidden_c, c), dim=0)
        self.store_log_probs = torch.cat((self.store_log_probs, log_probs))

        # store all else
        self.states = torch.cat((self.states, z), dim=0).detach()
        
    def activate_reset(self):
        # add to set
        self.hidden_h = torch.cat((self.hidden_h, self.store_hidden_h.detach()), dim=0)
        self.hidden_c = torch.cat((self.hidden_c, self.store_hidden_c.detach()), dim=0)
        self.log_probs = torch.cat((self.log_probs, self.store_log_probs.detach()), dim=0)
        self.reset_storage() # remove all else from buffer

    def save(self, mean, std):
        self.store_mean = torch.cat((self.store_mean, mean.unsqueeze(0)), dim=0)
        self.store_std = torch.cat((self.store_std, std.unsqueeze(0)), dim=0)

    def reward_prev(self, r_prev):
        self.rewards = torch.cat((self.rewards, r_prev), dim=0)

    def get_h_c(self, i):
        return self.hidden_h[-1], self.hidden_c[-1]

    def get_states(self):
        return self.states[-min(self.config.window_size, self.context_size):, :, :]

    def get_log_probs(self):
        return self.store_log_probs

    def get_rewards(self):
        return self.rewards[-min(self.config.window_size, self.context_size):, :, :]

# Analyzer and policy

class LSTMAnalyzer(nn.Module):
    def __init__(self, config):
        super(LSTMAnalyzer, self).__init__()
        self.config = config
        proj_size = config.action_dim*config.amt_actions
        self.rnn = nn.LSTM(config.x_dim, config.hidden_size, config.num_layers, dropout=config.dropout, proj_size=proj_size)
        if self.config.continuous:
            # mean, std heads
            self.mean_head = nn.Linear(proj_size, proj_size)
            self.std_head = nn.Linear(proj_size, proj_size)
        self.output_shape = (self.config.N, self.config.amt_actions, self.config.action_dim)
        # batch_first = False, so input is (seq, batch, feature)
        self.to(config.device) # move 
        self.to(config.dtype)

    def forward(self, history, prev_h_c, z):
        h, c = prev_h_c
        if history is None:# or True:
            output, (h, c) = self.rnn(z, (h, c)) # (1, N, action_dim*amt)
        else:
            #print(output)
            # oldest state - one to be booted off
            output, (h, c) = self.rnn(history[:1], (h, c))  # save next hidden states for window shift
            # rest of history
            h_, c_ = h, c
            if history.size()[0] > 1:
                output, (h_, c_) = self.rnn(history[1:], (h, c))
            # new state
            output, (_, _) = self.rnn(z, (h_, c_))
        if self.config.continuous:
            output = F.relu(output[-1, :, :])
            mean, std = self.mean_head(output), torch.exp(self.std_head(output))    # make std have non-wrong values
            # normalize for now
            mean = F.tanh(mean) * 1.99
            return (mean.view(self.output_shape), std.view(self.output_shape)), (h, c)
        else:
            out = output[-1, :, :].view(self.output_shape) # (1, N, action_dim*amt)
            return F.softmax(out, dim=2), (h, c) # last action - (1, N, amt_actions)

    def numel(self, only_trainable: bool = False):
        parameters = list(self.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        return sum(p.numel() for p in unique)

class Runner():

    def __init__(self, config):
        #super(Runner, self).__init__()
        self.analyzer = LSTMAnalyzer(config) 
        self.dataset = Dataset(config) # (L, N, Dims)
        self.conf = config
        self.eps = np.finfo(np.float64).eps.item()
        self.optimizer = optim.Adam(self.analyzer.parameters(), lr=config.lr)
        self.writer = SummaryWriter(config.dir, flush_secs=1, max_queue=2)
        self.writer_data = []
        self.eval = False # training = True
        print("params", self.analyzer.numel())

    def select_action(self, probs):
        # (N, action_dim*amt)
        if self.conf.continuous:
            mean, std = probs
            #mean = mean.view(self.analyzer.outpu(self.conf.N, self.conf.amt_actions, self.conf.action_dim))
            m = Normal(mean, std)
            self.dataset.save(mean, std)
        else:
            m = Categorical(probs)
        a = m.sample()
        return a, m.log_prob(a).unsqueeze(0)

    def add_prev_reward(self, reward):
        self.dataset.reward_prev(reward) #.unsqueeze(0))

    def policy(self, X): # given state
        z = self.conf.phi(X).unsqueeze(0)

        prev_h_c = self.dataset.get_h_c(-1)
        if self.dataset.context_size == 0:
            probs, (h, c) = self.analyzer(None, prev_h_c, z)
        else:
            history = self.dataset.get_states()
            probs, (h, c) = self.analyzer(history, prev_h_c, z)
        h, c = h.unsqueeze(0), c.unsqueeze(0)
        action, log_probs = self.select_action(probs) # get (N, amt), (N, amt)
        if not self.eval:
            self.dataset.add(z, log_probs, (h, c)) # S, log_probs, A pair add
        return action # action vector

    def reinforce(self, t): # "reinforce" prev t actions, due to scores of prev using hidden state prev, it doesn't work yet
       #print("\t\t\treinforcing actions!")
       self.analyzer.train() # this is training mode
       R = 0
       policy_loss = []
       returns = []
       # (L, N)
       for r in torch.flip(self.dataset.get_rewards()[-t:, :, :], (0,)):
           R = r + self.conf.gamma * R
           returns.insert(0, R)

       # normalize
       returns = torch.cat(returns)
       mean, std = self.dataset.mean_std_rewards
       mean, std = mean.unsqueeze(0), std.unsqueeze(0)
       returns = (returns - mean)/(std + self.eps) if self.dataset.context_size > 1 else returns

       # calculate reward with discount
       for log_prob, R, sigma in zip(self.dataset.get_log_probs()[-t:], returns, self.dataset.std):
           #pred = self.value(state)
           H = self.conf.beta * 1/2*torch.log(2*np.pi*np.e*sigma**2) * self.conf.continuous
           el = -log_prob * R - H
           policy_loss.append(el) #.sum((1, 2)) )
           #returns.

       # optimization
       self.optimizer.zero_grad()
       policy_loss = torch.cat(policy_loss).sum()
       policy_loss.backward()
       self.optimizer.step()

       self.dataset.activate_reset()

       # decay
       self.conf.beta = self.conf.beta * self.conf.beta_decay
       
       # add to tensorboard and data
       #self.writer_data.append(
               #np.array([mean.item(), std.item(), self.dataset.get_rewards().mean().item(), self.dataset.get_rewards().std().item()])
               #)

       self.writer.add_scalar("Rewards/mean_reward", mean.mean().item()) 
       self.writer.add_scalar("Rewards/std_reward", std.mean().item()) 
       self.writer.add_scalar("Rewards/past_window_mean_reward", self.dataset.get_rewards().mean(0).mean().item()) 
       self.writer.add_scalar("Rewards/past_window_std_reward", self.dataset.get_rewards().std(0).mean().item()) 

       self.checkpoint()

    def checkpoint(self):
        torch.save(self.analyzer.state_dict(), self.conf.dir + "/" + self.conf.checkpoint_dir)

    def load_model(self):
        self.analyzer.load_state_dict(torch.load(self.conf.dir + "/" + self.conf.checkpoint_dir))
        self.analyzer.eval()
        self.eval = True

    def over(self):
        # show data
        #for dat in self.writer_data:

        #plt.plot(self.writer_data)
        self.writer.close()
        print("Training over")
