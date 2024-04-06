import numpy as np
import gym

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
# from examples import IPenv
# import gymnasium as gym
from pilco.rewards import ExponentialReward
import tensorflow as tf
from gpflow import set_trainable

# from tensorflow import logging
np.random.seed(0)
# from examples import IPenv
from examples.IPenv import IPEnv
from utils import rollout, policy

import os
import pickle

def save_model(pilco, filename='best_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(pilco, f)
#使用Python的pickle模块来序列化PILCO对象并将其保存。
def load_model_if_exists(pilco, filename='best_model.pkl'):
    global best_reward
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            loaded_pilco = pickle.load(f)
            pilco.mgpr = loaded_pilco.mgpr
            pilco.controller = loaded_pilco.controller
            best_reward = loaded_pilco.best_reward
            print("Loaded model from", filename)
            print("Best reward from loaded model:", best_reward)
    else:
        best_reward = -np.inf


env = IPEnv()

# Initial random rollouts to generate a dataset
batch_timesteps =
total_timesteps = 4000
num_batches = total_timesteps // batch_timesteps

X_batches = []
Y_batches = []
for batch in range(num_batches):
    X_batch, Y_batch, _, _ = rollout(env=env, pilco=None, random=True, timesteps=batch_timesteps, render=True)
    X_batches.append(X_batch)
    Y_batches.append(Y_batch)
# 将整个数据收集过程分成了多个小批次,每个批次执行rollout函数,收集特定时间步的数据,减少了每次迭代所需的内存
state_dim = Y_batches[0].shape[1] # 计算了状态数据 Y 的列数，从而确定了状态空间的维度#读取矩阵第二维度的长度
control_dim = X_batches[0].shape[1] - state_dim  # 计算了控制输入数据 X 的列数减去状态空间的维度，从而确定了控制输入空间的维度（5-4）
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)

pilco = PILCO((X_batches[0], Y_batches[0]), controller=controller, horizon=40)

REWARD = []
#best_reward = -np.inf
X=X_batches[0]
Y=Y_batches[0]
load_model_if_exists(pilco, 'best_model.pkl')
for rollouts in range(1):
    for X_batch, Y_batch in zip(X_batches, Y_batches):# 遍历了所有批次，并在每个批次上独立更新和训练模型
        pilco.mgpr.set_data((X_batch, Y_batch))  # 使用当前批次数据更新模型,确保模型在每个批次上都使用最新的数据进行训练

        pilco.optimize_models() # 优化高斯过程
        pilco.optimize_policy() # 优化控制策略

        X_new, Y_new, _, reward_ges = rollout(env=env, pilco=pilco, timesteps=batch_timesteps, render=True)
        REWARD.append(reward_ges)

ji    X = np.vstack((X, X_new))
    Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))

    if reward_ges > best_reward:#如果本次迭代的reward超过了原先的reward
        best_reward = reward_ges
        pilco.best_reward = best_reward
        save_model(pilco, 'best_model.pkl')



import matplotlib.pyplot as plt

plt.figure()
plt.plot(REWARD)
plt.show()

print(REWARD)