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

# # 导入Spinning Up的run模块
# import spinup
# from spinup.run import run

# if __name__ == "__main__":
#     # 定义训练参数
#     locals().update({
#         'algo': 'ppo',  # 使用的算法
#         'env': 'HuaRongDao-v0',  # 环境ID
#         'exp_name': 'hrd_test',  # 实验名称
#         'hid': [32, 32],  # 隐藏层大小
#         'gamma': 0.999,  # 折扣因子
#         'seed': 0,  # 随机种子
#         'cpu': 4,  # 使用的CPU核心数
#         'steps_per_epoch': 4000,  # 每个epoch的步数
#         'epochs': 50,  # 训练的epochs数量
#         'save_freq': 10,  # 每隔多少epochs保存一次模型
#         'render': False,  # 是否渲染环境
#         'log_dir': './logs',  # 日志保存路径
#     })

#     # 运行Spinning Up的训练命令
#     run(**locals())

# 导入Spinning Up的算法函数
from spinup import ppo_pytorch as ppo
import gym

if __name__ == "__main__":
    # 定义训练参数
    env_fn = lambda: gym.make('HuaRongDao-v0')
    ac_kwargs = dict(hidden_sizes=[32, 32], activation=torch.nn.ReLU)
    logger_kwargs = dict(output_dir='./logs/hrd_test', exp_name='hrd_test')

    # 调用PPO算法
    ppo(env_fn=env_fn,
        ac_kwargs=ac_kwargs,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.999,
        logger_kwargs=logger_kwargs)