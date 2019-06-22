import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size):
        super(Policy,self).__init__()

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(num_outputs, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(num_outputs, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(num_outputs, num_outputs)

        self.base = nn.Linear(obs_shape[0], hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.action_head = nn.Linear(hidden_size, 2)

    def forward(self,x):
        x = F.relu(self.base(x))
        v = self.value_head(x)
        a_latent = self.action_head(x)
        return v, a_latent

    def act(self,latent):
        dist = self.dist(latent)
        action = dist.sample()
        log_prob = dist.log_probs(action)
        return action, log_prob

def generateEpisode(env, policy, args):
    rewards = []
    values = []
    action_log_probs = []

    obs = env.reset()

    for i in range(10000):
        obs = torch.from_numpy(obs).float()
        v, a = policy(obs)
        action, log_prob = policy.act(a)

        values.append(v)
        action_log_probs.append(log_prob)

        obs,reward,done,_ = env.step(action.item())

        if args.render:
            env.render()

        rewards.append(reward)

        if done:
            break

    return rewards, values, action_log_probs


def updatePolicy(optimizer, rewards, values, action_log_probs, args):
    R = 0
    returns = []

    for r in rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0,R)

    returns = torch.tensor(returns)
    returns = (returns - torch.mean(returns))/ torch.std(returns)

    l_critic = []
    l_actor = []

    values = torch.stack(values)
    values = values.view(-1)
    adv = returns - values
    l_critic = F.smooth_l1_loss(returns, values)
    l_actor = - torch.matmul(torch.stack(action_log_probs).view(-1), adv) / len(rewards)


    optimizer.zero_grad()

    loss = l_critic + l_actor
    loss.backward()
    optimizer.step()

def main():
    parser = argparse.ArgumentParser(description="actor_critic")
    parser.add_argument('--env_name', type = str, default="CartPole-v0")
    parser.add_argument('--seed', type = int, default=1)
    parser.add_argument('--render', type = bool, default=True)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--gamma', type=float, default= 0.99)

    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(args.seed)

    policy = Policy(env.observation_space.shape,
                     env.action_space,128)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    for t in range(100000):
        rewards, values, action_log_probs = generateEpisode(env,policy,args)
        updatePolicy(optimizer, rewards, values, action_log_probs, args)

        if t % 10 == 0:
            print("rewards: {}".format(np.sum(np.asarray(rewards))))

if __name__ == '__main__':
    main()
