import torch
from env import *
from reinforce import PendulumNet
from tqdm import trange

e = PendulumEnv(torch.float64, torch.device("cuda"), N=100, Net=PendulumNet)
s = e.reset()
for i in trange(1000):
    s, r_prev, r = e.step(torch.zeros((100, 1, 1), dtype=torch.float64, device=torch.device("cuda")))
print(r.mean())

