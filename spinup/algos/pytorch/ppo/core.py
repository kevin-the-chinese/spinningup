import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def mlp_act(sizes, activation, output_activation=nn.Softmax):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act(dim=-1) if act == nn.Softmax else act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp_act([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(128, 64, 128, 64, 64), activation=nn.ReLU, 
                 load_path=None):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

        # Load model if load_path is provided
        if load_path:
            self.load_model(load_path)
            # self.load_original_params(load_path)

    def save_model(self, save_path):
        # Save both policy and value network
        torch.save(self.pi.state_dict(), f"{save_path}_pi.pth")
        torch.save(self.v.state_dict(), f"{save_path}_v.pth")

    def load_model(self, load_path):
        # Load both policy and value network
        self.pi.load_state_dict(torch.load(f"{load_path}_pi.pth", map_location=torch.device('cpu')))
        self.v.load_state_dict(torch.load(f"{load_path}_v.pth", map_location=torch.device('cpu')))
    # Load original model parameters
    def load_original_params(self, load_path):
        # Load original model parameters
        pi_original_state_dict = torch.load(f"{load_path}_pi.pth", map_location=torch.device('cpu'))
        v_original_state_dict = torch.load(f"{load_path}_v.pth", map_location=torch.device('cpu'))
        
        # Get current model's state dictionaries
        pi_new_state_dict = self.pi.state_dict()
        v_new_state_dict = self.v.state_dict()

        # Copy parameters from the original model to the new model if shapes match
        for key in pi_original_state_dict:
            if key in pi_new_state_dict and pi_original_state_dict[key].shape == pi_new_state_dict[key].shape:
                pi_new_state_dict[key] = pi_original_state_dict[key]

        for key in v_original_state_dict:
            if key in v_new_state_dict and v_original_state_dict[key].shape == v_new_state_dict[key].shape:
                v_new_state_dict[key] = v_original_state_dict[key]

        # Load the updated state dictionaries into the new model
        self.pi.load_state_dict(pi_new_state_dict)
        self.v.load_state_dict(v_new_state_dict)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]