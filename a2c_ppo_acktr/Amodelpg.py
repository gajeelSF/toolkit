import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class APolicy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(APolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = MLPBase

        self.hidden_size  = 256


        if action_space.__class__.__name__ == "Discrete":
            num_inputs = action_space.n + obs_shape[0] * 2
            #num_inputs =  obs_shape[0]
            num_outputs = action_space.n
            self.dist = Categorical(self.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_inputs = action_space.shape[0] + obs_shape[0] * 2
            #num_inputs = obs_shape[0]

            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_inputs = action_space.shape[0] + obs_shape[0] * 2
            #num_inputs = obs_shape[0]
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.hidden_size, num_outputs)
        else:
            raise NotImplementedError

        self.base = base(num_inputs,num_outputs)
        #self.base = base(obs_shape[0],num_outputs)
    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def adapt(self, inputs, deterministic=False):
        actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return  action, action_log_probs

    def evaluate_actions(self, inputs, action):
        actor_features = self.base(inputs)

        #print(actor_features.shape)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        return action_log_probs

class MLPBase(nn.Module):
    def __init__(self, num_inputs,num_outputs, hidden_size=256):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(0.2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.ReLU())

        self.train()

    def forward(self, inputs):
        x = inputs
        hidden_actor = self.actor(x)

        return hidden_actor
