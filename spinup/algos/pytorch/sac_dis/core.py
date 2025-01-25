import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from gym.spaces import Box, Discrete

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

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# LOG_STD_MAX = 2
# LOG_STD_MIN = -20

# class SquashedGaussianMLPActor(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
#         self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.act_limit = act_limit

#     def forward(self, obs, deterministic=False, with_logprob=True):
#         net_out = self.net(obs)
#         mu = self.mu_layer(net_out)
#         log_std = self.log_std_layer(net_out)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)

#         # Pre-squash distribution and sample
#         pi_distribution = Normal(mu, std)
#         if deterministic:
#             # Only used for evaluating policy at test time.
#             pi_action = mu
#         else:
#             pi_action = pi_distribution.rsample()

#         if with_logprob:
#             # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
#             # NOTE: The correction formula is a little bit magic. To get an understanding 
#             # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
#             # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
#             # Try deriving it yourself as a (very difficult) exercise. :)
#             logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
#             logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
#         else:
#             logp_pi = None

#         pi_action = torch.tanh(pi_action)
#         pi_action = self.act_limit * pi_action

#         return pi_action, logp_pi


# class MLPQFunction(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

#     def forward(self, obs, act):
#         q = self.q(torch.cat([obs, act], dim=-1))
#         return torch.squeeze(q, -1) # Critical to ensure q has right shape.

# pro
class MLPQFunctionDis(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        return self.q(obs)


def CategoricalSample(q, deterministic=False):
    try:
        pi = Categorical(logits=q)
        if deterministic:
            a = torch.argmax(pi.probs, dim=-1)
        else:
            a = pi.sample()
        logp_a = pi.log_prob(a)
        return a.item(), logp_a
    except Exception as e:
        print("Error in CategoricalSample:")
        print("q_values:", q)
        print("Exception:", e)
        raise  # 重新抛出异常以便进一步处理    

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, load_path=None):
        super().__init__()

        # pro: discreate
        obs_dim = observation_space.shape[0]
        self.q1 = MLPQFunctionDis(obs_dim, action_space.n, hidden_sizes, activation)
        self.q2 = MLPQFunctionDis(obs_dim, action_space.n, hidden_sizes, activation)
        # Load model if load_path is provided
        if load_path:
            self.load_model(load_path)
            # self.load_original_params(load_path)

    def save_model(self, save_path):
        # Save both policy and value network
        torch.save(self.q1.state_dict(), f"{save_path}_q1.pth")
        torch.save(self.q2.state_dict(), f"{save_path}_q2.pth")

    def load_model(self, load_path):
        # Load both policy and value network
        self.q1.load_state_dict(torch.load(f"{load_path}_q1.pth", map_location=torch.device('cpu')))
        self.q2.load_state_dict(torch.load(f"{load_path}_q2.pth", map_location=torch.device('cpu')))
    # Load original model parameters
    def load_original_params(self, load_path):
        # Load original model parameters
        q1_original_state_dict = torch.load(f"{load_path}_q1.pth", map_location=torch.device('cpu'))
        q2_original_state_dict = torch.load(f"{load_path}_q2.pth", map_location=torch.device('cpu'))
        
        # Get current model's state dictionaries
        q1_new_state_dict = self.q1.state_dict()
        q2_new_state_dict = self.q2.state_dict()

        # Copy parameters from the original model to the new model if shapes match
        for key in q1_original_state_dict:
            if key in q1_new_state_dict and q1_original_state_dict[key].shape == q1_new_state_dict[key].shape:
                q1_new_state_dict[key] = q1_original_state_dict[key]

        for key in q2_original_state_dict:
            if key in q2_new_state_dict and q2_original_state_dict[key].shape == q2_new_state_dict[key].shape:
                q2_new_state_dict[key] = q2_original_state_dict[key]

        # Load the updated state dictionaries into the new model
        self.q1.load_state_dict(q1_new_state_dict)
        self.q2.load_state_dict(q2_new_state_dict)


    def act(self, obs, deterministic=False):
        try:
            with torch.no_grad():
                q1_values = self.q1(obs)
                q2_values = self.q2(obs)
                q_values = torch.min(q1_values, q2_values)
                return CategoricalSample(q_values, deterministic)
        except Exception as e:
            print("Error in act:")
            print("q1_values:", q1_values)
            print("q2_values:", q2_values)
            print("q_values:", q_values)
            print("Exception:", e)
            raise  # 重新抛出异常以便进一步处理
