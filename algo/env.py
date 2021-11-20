import torch
import torch.optim as optim
import gym
import numpy as np
from os import path

# pendulum env in pytorch (much faster)
class PendulumEnv():
    def __init__(self, dtype, device, Net, trajectory=40, lr=1e-4, N=1, g=10.0, pick=0):
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.N = N
        self.viewer = None
        self.cnt = 0
        self.trajectory = trajectory

        high = torch.tensor([1.0, 1.0, self.max_speed], device=self.device, dtype=self.dtype)

        # space dims
        self.state_space = 3
        self.action_space = 1

        self.pick = pick

        self.amt = 1

        # net
        self.net = Net(action_space=self.action_space, state_space=self.state_space) # network for reward
        self.net.to(device)
        self.net.to(dtype)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)


        #self.action_space = spaces.Box(
        #    low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        #)
        #self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)


    def wrap(self, t):
        return torch.tensor(t, device=self.device, dtype=self.dtype)

    def predict(self, s, a, train=True):
        # TODO: train vs test
        #if train:
        #    self.net.train()
        #else:
        #    self.net.eval()
        X = torch.cat((s, a), dim=1)
        s_t = self.net(X)
        return s_t

    def update(self, s_pred, s_actual):
        self.optimizer.zero_grad()
        #print((s_pred).shape)
        loss_rew = ((s_pred - s_actual)**2).sum(1).unsqueeze(1)
        loss = loss_rew.sum()
        loss.backward()
        self.optimizer.step()
        return loss_rew.detach().clone()

    def get_reward(self, obs, a):
        s_pred = self.predict(obs, a) 
        pred_reward = self.update(s_pred, obs)
        return pred_reward

    def step(self, u):
        self.cnt += 1
        # predict first
        u = u.squeeze(2)
        obs = self._get_obs()
        s_pred = self.predict(obs.detach(), u.clone().detach())

        th, thdot = self.state[:, :1], self.state[:, 1:]
        #th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = torch.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u[self.pick].item() # for rendering, chose random one
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = torch.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = torch.cat((newth, newthdot), dim=1)
        r_env = -costs
        obs = self._get_obs()
        r_pred = self.update(s_pred, obs)
        #r_pred = self.get_reward(obs, u)
        return obs, r_pred, r_env

    def reset(self, flag=True):
        high = np.array([np.pi, 1])
        #self.state = np.random.uniform(low=-high, high=high)
        #self.th = torch.rand((self.N, 1), device=self.device, dtype=self.dtype) * (high[0]*2) - high[0]
        self.th = torch.zeros((self.N, 1), device=self.device, dtype=self.dtype)
        self.thdot = torch.zeros((self.N, 1), device=self.device, dtype=self.dtype)
        #self.thdot = torch.rand((self.N, 1), device=self.device, dtype=self.dtype) * (high[1]*2) - high[1]
        self.state = torch.cat((self.th, self.thdot), dim=1)

        #self.state = torch.tensor(self.state, device=self.device, dtype=self.dtype)
        #self.state = self.wrap(self.state)
        self.last_u = None
        if flag:
            return self._get_obs()

    def _get_obs(self):
        if self.trajectory is not None and self.cnt % self.trajectory and self.cnt != 0:
            self.reset(False)

        theta, thetadot = self.state[:, :1], self.state[:, 1:]
        out = torch.cat((torch.cos(theta), torch.sin(theta), thetadot), dim=1) # * 10 # scale factor
        return out

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[self.pick, 0].item() + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class RandEnv():
    # rand env
    def __init__(self, continuous, device, dtype):
        self.state_space = 3
        self.action_space = 4
        self.amt = 2
        self.dtype = dtype
        self.device = device

    def rand_state(self):
        return torch.rand(self.state_space, dtype=self.dtype, device=self.device)

    def reset(self):
        return self.rand_state()

    def step(self, action):
        assert tuple(action.size()) == (2, 4)
        s = self.rand_state()
        r = torch.rand(1, dtype=self.dtype, device=self.device)
        return s, r

if __name__ == "__main__":
    env = PendulumEnv(dtype=torch.float32, device="cuda")
    env.reset()
    for i in range(10000):
        s, r = env.step(torch.tensor([1], dtype=torch.float32).cuda())
        env.render()
