import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy


class Actor(nn.Module):
    def __init__(self,obs_shape, action_space, hidden_size):
        super(Actor,self).__init__()
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]

        self.fc1 = nn.Linear(obs_shape[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_outputs)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class Critic(nn.Module):
    def __init__(self,obs_shape, hidden_size):

        self.fc1 = nn.Linear(obs_shape[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class ReplayBuffer():
    def init(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_state = []
    def append(self, s, a, r, sp):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_state.append(sp)
    def sample(self, n):
        

    def length(self):
        return len(states)



def updatePolicy(actor, target_actor, actor_optimizer, critic, target_critic, critic_optimizer, replay_buffer):



def main():
    parser = argparse.ArgumentParser(description="ddpg")
    parser.add_argument('--env_name', type = str, default="HalfCheetah-v2")
    parser.add_argument('--seed', type = int, default=1)
    parser.add_argument('--render', type = bool, default=True)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--gamma', type=float, default= 0.99)
    parser.add_argument('batch_size', type=int, default=32)

    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(args.seed)

    critic = Critic(env.observation_space.shape,128)
    target_critic = copy.deepcopy(critic)

    actor = Actor(env.observation_space.shape,env.action_space,128)
    target_actor = copy.deepcopy(actor)

    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)

    replay_buffer = ReplayBuffer()

    for eps in range(10000):

        obs = env.reset()
        while True:
            obs_tensor = torch.from_numpy(obs).float()
            action = actor(obs_tensor)

            next_obs,reward,done,_ = env.step(action.item())
            replay_buffer.append(obs,action,reward,next_obs)

            if replay_buffer.length >= args.batch_size:
                updatePolicy(actor, target_actor, actor_optimizer, critic, target_critic,
                            critic_optimizer, replay_buffer)

            if done:
                break
            obs = next_obs
