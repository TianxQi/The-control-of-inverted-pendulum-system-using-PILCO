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
from examples.IPenv_stab import IPEnv
from utils import rollout, policy

import os
import pickle
def save_data(X, Y, filename='data_stab_try.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((X, Y), f)
def load_data(filename='data.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_data(X, Y, filename='data_try05.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((X, Y), f)
def save_pilco(pilco, filename='pilco_model_try.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(pilco, f)
    print("PILCO model saved to", filename)
def save_model(pilco, filename='best_model_try.pkl'):
    model_data = {
        'mgpr': pilco.mgpr,
        'controller': pilco.controller
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model data saved to", filename)
def load_model_if_exists(pilco, filename='best_model_try.pkl'):
    global best_reward
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            pilco.mgpr = model_data['mgpr']  # Load the saved model parts
            pilco.controller = model_data['controller']
            print("Loaded model from", filename)

def save_best_reward(best_reward, filename='best_reward_try.pkl'):

    with open(filename, 'wb') as f:
        pickle.dump(best_reward, f)
    print("Best reward saved to", filename)
    print('Best reward ist ',best_reward)

def load_best_reward(filename='best_reward_try.pkl'):
    global best_reward
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            best_reward = pickle.load(f)
            print("Loaded best reward from", filename)
            print('Best reward ist ',best_reward)
            return best_reward
    else:
        print("No best reward found.")
        print('Best reward ist ', -np.inf)
        return -np.inf
X, Y = load_data()

env = IPEnv()
# 初始化PILCO模型
state_dim = Y.shape[1]  # 计算了状态数据 Y 的列数，从而确定了状态空间的维度#读取矩阵第二维度的长度
control_dim = X.shape[1] - state_dim  # 计算了控制输入数据 X 的列数减去状态空间的维度，从而确定了控制输入空间的维度（5-4）
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)

pilco = PILCO((X, Y), controller=controller, horizon=40)

REWARD = []
#load_model_if_exists(pilco, 'best_model_try.pkl')
best_reward = load_best_reward()
for rollouts in range(5):

    pilco.optimize_models()  # 优化高斯过程
    pilco.optimize_policy()  # 优化控制策略

    X_new, Y_new, _, reward_ges = rollout(env=env, pilco=pilco, timesteps=100, render=True)  # 用当前策略模拟环境450个时间步
    REWARD.append(reward_ges)

    X = np.vstack((X, X_new))  # 新数据添加到现有数据集
    Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))
    if reward_ges > best_reward:  # 如果本次迭代的reward超过了原先的reward
        print('The best reward this time:', reward_ges)
        print('The top reward in history: ', best_reward)
        best_reward = reward_ges
        # pilco.best_reward = best_reward
        save_model(pilco, 'best_model_try.pkl')
        save_best_reward(best_reward, filename='best_reward_try.pkl')
        save_pilco(pilco,'pilco_model_try.pkl')
        save_data(X,Y,filename='data_stab_try')
    print('The RUN times:', rollouts + 1)

# import matplotlib.pyplot as plt#绘制reward图像
# plt.figure()
# plt.plot(REWARD)
# plt.show()
print(REWARD)
# def load_pilco(filename='pilco_model.pkl'):
#     if os.path.isfile(filename):
#         with open(filename, 'rb') as f:
#             pilco = pickle.load(f)
#             print("Loaded PILCO model from", filename)
#             return pilco
#     else:
#         print("No PILCO model found at", filename)
#         return None
#
# pilco = load_pilco('pilco_model.pkl')
# if pilco is None:
#     # 初始化PILCO模型
#     pilco = PILCO((X, Y), controller=controller, horizon=40)