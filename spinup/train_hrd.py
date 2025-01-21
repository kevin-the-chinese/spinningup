# train_hrd.py
import sys
import os

from spinup.user_config import DEFAULT_BACKEND
from spinup.utils.run_utils import ExperimentGrid
from spinup.utils.serialization_utils import convert_json
import argparse
import gym
import json
import os, subprocess, sys
import os.path as osp
import string
import tensorflow as tf
import torch
from copy import deepcopy
from textwrap import dedent

# 添加包含hrd.py的目录到Python模块路径中
sys.path.append('/workspaces/spinningup/spinup/usr_envs')
import hrd

# 导入Spinning Up的算法函数
from spinup import ppo_pytorch as ppo
import gym

if __name__ == "__main__":
    # 定义训练参数
    env_fn = lambda: gym.make('HuaRongDao-v0')
    ac_kwargs = dict(hidden_sizes=[128, 64, 128], activation=torch.nn.ReLU)
    logger_kwargs = dict(output_dir='./logs/hrd_test', exp_name='hrd_test')

    # 调用PPO算法
    ppo(env_fn=env_fn,
        ac_kwargs=ac_kwargs,
        steps_per_epoch=4000,
        epochs=500,
        gamma=0.98,
        logger_kwargs=logger_kwargs)