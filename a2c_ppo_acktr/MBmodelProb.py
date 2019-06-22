import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Model, self).__init__()

        self.hidden_size = 128

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = MLPBase
        if action_space.__class__.__name__ == "Discrete":
            num_inputs = action_space.n + obs_shape[0]
        elif action_space.__class__.__name__ == "Box":
            num_inputs = action_space.shape[0] + obs_shape[0]
        elif action_space.__class__.__name__ == "MultiBinary":
            num_inputs = action_space.shape[0] + obs_shape[0]
        else:
            raise NotImplementedError

        num_outputs = obs_shape[0]
        self.dist = DiagGaussian(self.hidden_size, num_outputs)
        self.base = base(num_inputs,obs_shape[0])

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def predict(self, inputs, deterministic=False):
        hidden = self.base(inputs)
        dist = self.dist(hidden)
        if deterministic:
            sp = dist.mode()
        else:
            sp = dist.sample()
        return  sp

    def evaluate(self, inputs, y):
        hidden = self.base(inputs)
        dist = self.dist(hidden)
        log_prob = dist.log_probs(y)
        return log_prob


class MLPBase(nn.Module):
    def __init__(self, num_inputs,num_outputs, hidden_size=128):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.train()

    def forward(self, inputs):
        x = inputs
        hidden_actor = self.actor(x)

        return hidden_actor
