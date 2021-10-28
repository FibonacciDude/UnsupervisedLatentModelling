import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils import *
from env import *
from tqdm import trange
import gym


torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

dtype = torch.float64
device = torch.device("cuda")

class PendulumNet(nn.Module):
    def __init__(self, action_space, state_space):
        super(PendulumNet, self).__init__()
        hidden = 128
        self.fc1 = nn.Linear(action_space+state_space, hidden) # action and state -> hidden
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, state_space)

    def forward(self, X):
        max_speed = 8 # pendulum
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        # normalize
        X[:, 2:] = F.tanh(X[:, 2:])
        X[:, -1] = F.tanh(X[:, -1]) * max_speed
        return X

continuous = True
N = 100
pick = 10
env = PendulumEnv(device=device, dtype=dtype, N=N, Net=PendulumNet, lr=5e-4, pick=pick)
x_dim = env.state_space
action_dim = env.action_space
amt = env.amt
hidden_size = 64
num_layers = 2
config = Config(x_dim,
        action_dim,
        N,
        hidden_size,
        num_layers,
        trajectory=10,
        dropout=0,
        gamma=.9,
        #window_size=100,
        window_size=100,
        beta=.1,
        beta_decay=.5,
        amt_actions=amt,
        device=device,
        dtype=dtype,
        lr=1e-4,
        continuous=continuous,
        dir="pendulum_data",
        checkpoint_dir="model.pth",
        phi=lambda x:x)


"""
policy(X) # action
reinforce(optimizer, t) # rl
add_prev_reward(reward)
"""

def training(run_for, seed, intrinsic=True):
    runner = Runner(config)
    # change as required
    # remember that action has to include sequence length L
    # TODO: required_grad = True may not do the correct thing, check
    #envs = [PendulumEnv(device=device, dtype=dtype) for i in range(N)]
    #S = torch.tensor([e.reset() for e in envs], device=device, dtype=dtype)
    s = env.reset()
    R = []  # env rewards
    R_int = []
    for i in trange(run_for):
        action = runner.policy(s) # take action, forward
        s, r_prev, r_env = env.step(action) # see effects
        if intrinsic:
            r = r_prev
        else:
            r = r_env
        runner.add_prev_reward(r.unsqueeze(0)) # update reward for last action (this one), remember that this is for all N envs
        R_int.append(r_prev.cpu().numpy())
        R.append(r_env.cpu().numpy())
        #print(i % config.trajectory)
        if i % config.trajectory == 0:
            runner.reinforce(config.trajectory+1) # reinforce it
    NN = 20
    # env rewards
    R = np.array(R)
    R_avg = np.array([arr.mean() for arr in np.array_split(R, NN)])
    R_std = np.array([arr.std() for arr in np.array_split(R, NN)])

    # intrinsic rewards
    R_int = np.array(R_int)
    R_int_avg = np.array([arr.mean() for arr in np.array_split(R_int, NN)])
    R_int_std = np.array([arr.std() for arr in np.array_split(R_int, NN)])
    
    # output
    rewards = runner.dataset.get_rewards().cpu().numpy()
    #if intrinsic:
    #    print("mean/std *intrinsic* reward last context", rewards.mean(), "/", rewards.std())
    #context = 30
    #print("mean/std env reward last context (only %d steps back)" % (context), R[-context:].mean(), "/", R[-context].std())
    #RR = (R_avg, R_std) if not intrinsic else (R_int_avg, R_int_std)
    RR = (R_avg, R_std)
    prefix = "data/"
    np.save(prefix + str(seed)+"_"+("intrinsic" if intrinsic else "extrinsic") + "_mean", RR[0])
    np.save(prefix + str(seed)+"_"+("intrinsic" if intrinsic else "extrinsic")+"_std", RR[1])
    print("env rewards mean", R_avg)
    print("env rewards std", R_std)
    print("intr reward mean", R_int_avg)
    print("intr reward std", R_int_std)
    print("------------------------------")

    runner.over()

def testing(render=True):
    runner = Runner(config)
    s = env.reset()
    runner.load_model()
    for i in range(100):
        if render:
            env.render()
        action = runner.policy(s) # take action, forward
        s, r_prev, r_env = env.step(action) #.unsqueeze(0)) # see effects
        runner.add_prev_reward(r_prev.unsqueeze(0)) # update reward for last action (this one), remember that this is for all N envs

if __name__ == "__main__":
    seeds = [39, 42890, 3424, 23232]
    #seeds = []
    run_for = 200
    for intr in [True, False]:
        for seed in seeds:
            print("------------------------------")
            print("intrinsic?", intr, "\tseed:", seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            training(run_for, intrinsic=intr, seed=seed)
            #testing(False)
