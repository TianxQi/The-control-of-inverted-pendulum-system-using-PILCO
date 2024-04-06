import numpy as np
#import gym

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController

import gymnasium as gym

import tensorflow as tf
from gpflow import set_trainable

# from tensorflow import logging
np.random.seed(0)
# from examples import IPenv
from examples.IPEnv_SAC import IPEnv_SAC
from utils import rollout, policy

import os
import pickle

from stable_baselines3 import SAC

env = IPEnv_SAC

model = SAC("MlpPolicy", env, verbose=1,
              learning_rate=0.01,  # 学习率
            #  buffer_size=1000000,   # 重放缓冲区大小
            #  batch_size=1024,        # 批量大小
              gamma=0.9,            # 折扣因子
            #  tau=0.007,             # 目标网络的软更新参数
              ent_coef='auto_0.1',       # 熵系数，'auto'让算法自动调整
            #  target_update_interval=1, # 目标网络的更新间隔
            #  train_freq=(1, "episode"), # 训练频率
            #  gradient_steps=-1,     # 每个环境步骤后进行的梯度步数，-1表示仅在每个环境结束时更新
            # #action_noise=action_noise, # 动作噪声
            )   # 使用模型预测动作
model.learn(total_timesteps=1600, log_interval=4)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:#terminated or truncated:
        obs = env.reset()