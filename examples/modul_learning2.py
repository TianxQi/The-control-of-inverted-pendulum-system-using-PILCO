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
# def save_model(pilco, filename='model_data.pkl'):
#     model_data = {
#         'mgpr': pilco.mgpr,
#         'controller': pilco.controller
#     }
#     with open(filename, 'wb') as f:
#         pickle.dump(model_data, f)
#     print("Model data saved to", filename)
#
# def save_best_reward(best_reward, filename='best_reward.pkl'):
#     with open(filename, 'wb') as f:
#         pickle.dump(best_reward, f)
#     print("Best reward saved to", filename)
#     print('Best reward ist ',best_reward)
#
# def load_best_reward(filename='best_reward.pkl'):
#     if os.path.isfile(filename):
#         with open(filename, 'rb') as f:
#             best_reward = pickle.load(f)
#             print("Loaded best reward from", filename)
#             print('Best reward ist ',best_reward)
#             return best_reward
#     else:
#         print("No best reward found.")
#         print('Best reward ist ', -np.inf)
#         return -np.inf
#
# #best_reward = load_best_reward()
# # ... 在训练过程中更新 best_reward ...
# #save_best_reward(best_reward)
# #使用Python的pickle模块来序列化PILCO对象并将其保存。
# def load_model_if_exists(pilco, filename='best_model.pkl'):
#     global best_reward
#     if os.path.isfile(filename):
#         with open(filename, 'rb') as f:
#             model_data = pickle.load(f)
#             pilco.mgpr = model_data['mgpr']  # Load the saved model parts
#             pilco.controller = model_data['controller']
#             print("Loaded model from", filename)


def save_REWARD(REWARD, filename='reward_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(REWARD, f)
    print("REWARD saved to", filename)


X, Y = load_data()
env = IPEnv()
# 初始化PILCO模型
state_dim = Y.shape[1]  # 计算了状态数据 Y 的列数，从而确定了状态空间的维度#读取矩阵第二维度的长度
control_dim = X.shape[1] - state_dim  # 计算了控制输入数据 X 的列数减去状态空间的维度，从而确定了控制输入空间的维度（5-4）
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)

pilco = PILCO((X, Y), controller=controller, horizon=40)

REWARD = []
#best_reward = -np.inf # 负无穷的初始变量
#load_model_if_exists(pilco, 'best_model.pkl') # 加载模型
#best_reward = load_best_reward()

for rollouts in range(8):

    pilco.optimize_models()  # 优化高斯过程
    pilco.optimize_policy()  # 优化控制策略

    X_new, Y_new, _, reward_ges = rollout(env=env, pilco=pilco, timesteps=100, render=True)  # 用当前策略模拟环境450个时间步
    REWARD.append(reward_ges)

    X = np.vstack((X, X_new))  # 新数据添加到现有数据集
    Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))  # 用新的组合数据集更新模型
    print('已运行',rollouts+1)
    # if reward_ges > best_reward:  # 如果本次迭代的reward超过了原先的reward
    #     print('The best reward this time:', reward_ges)
    #     print('The top reward in history: ', best_reward)
    #     best_reward = reward_ges
    #
    #     # pilco.best_reward = best_reward
    #     #save_model(pilco, 'best_model.pkl')
    #     #save_best_reward(best_reward, filename='best_reward.pkl')
    # save_REWARD(REWARD, f'reward_data_{rollouts}.pkl')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(REWARD)
plt.show()

print(REWARD)
