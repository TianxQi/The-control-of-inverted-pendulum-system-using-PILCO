import numpy as np
import gym

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController

# import gymnasium as gym

import tensorflow as tf
from gpflow import set_trainable

# from tensorflow import logging
np.random.seed(0)
# from examples import IPenv
from examples.IPenv import IPEnv
from utils import rollout, policy

import os
import pickle

def load_data(filename='data.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_REWARD(REWARD, filename='reward_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(REWARD, f)
    print("REWARD saved to", filename)
def save_model(pilco, best_reward, filename='best_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({'pilco': pilco, 'best_reward': best_reward}, f)

def load_model_if_exists(pilco, filename='best_model.pkl'):
    global best_reward
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            pilco.mgpr = data['pilco'].mgpr
            pilco.controller = data['pilco'].controller
            best_reward = data['best_reward']
            print("Loaded model from", filename)
            print("Best reward from loaded model:", best_reward)
    else:
        best_reward = -np.inf


X, Y = load_data()
env = IPEnv()
    # 初始化PILCO模型
state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)
pilco = PILCO((X, Y), controller=controller, horizon=40)

REWARD = []
# best_reward = -np.inf # 负无穷的初始变量
load_model_if_exists(pilco, 'best_model.pkl')  # 加载模型

for rollouts in range(2):

    pilco.optimize_models()  # 优化高斯过程
    pilco.optimize_policy()  # 优化控制策略

    X_new, Y_new, _, reward_ges = rollout(env=env, pilco=pilco, timesteps=4500, render=True)  # 用当前策略模拟环境450个时间步
    REWARD.append(reward_ges)

    X = np.vstack((X, X_new))  # 新数据添加到现有数据集
    Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))  # 用新的组合数据集更新模型

    if reward_ges > best_reward:  # 如果本次迭代的reward超过了原先的reward
        best_reward = reward_ges
        # pilco.best_reward = best_reward
        save_model(pilco, best_reward, 'best_model.pkl')

    save_REWARD(REWARD, f'reward_data_{rollouts + 100 * k}.pkl')

import matplotlib.pyplot as plt

plt.figure()
plt.plot(REWARD)
plt.show()
