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

env = IPEnv()
# Initial random rollouts to generate a dataset
X, Y, _, _ = rollout(env=env, pilco=None, random=True, timesteps=50, render=True)
for i in range(1, 5):  # 迭代 4 次
    X_, Y_, _, _ = rollout(env=env, pilco=None, random=True, timesteps=50, render=True)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))
#生成初始的数据集，在环境中多次执行随机轨迹，以收集足够的状态-控制输入数据对，这将用于后续的PILCO模型的训练
#PILCO 模型被初始化。它包括一个 Gaussian Process（GP）模型（mgpr）、一个控制器（controller）和一个奖励模型（默认为指数奖励）。这个模型将在后续的循环中进行优化。
state_dim = Y.shape[1]  # 计算了状态数据 Y 的列数，从而确定了状态空间的维度#读取矩阵第二维度的长度
control_dim = X.shape[1] - state_dim  # 计算了控制输入数据 X 的列数减去状态空间的维度，从而确定了控制输入空间的维度（5-4）
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)
# 1个RBF (Radial Basis Function) 控制器被初始化.
# state_dim：之前计算的状态空间的维度。
# control_dim：之前计算的控制输入空间的维度。
# num_basis_functions=10：RBF控制器通常使用一组基函数来建模控制策略。这个参数指定了基函数的数量，这里设置为10个。
# controller = LinearController(state_dim=state_dim, control_dim=control_dim)
#R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]),W=np.diag([1,1,2,1])
pilco = PILCO((X, Y), controller=controller, horizon=40)
# Example of user provided reward function, setting a custom target state
# R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
# pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)
REWARD = []

#pilco.load_model_if_exists()
#controller_old = pilco.load_model()
#pilco.controller.models = controller_old.models
#pilco.controller.optimizers = controller_old.optimizers
for rollouts in range(2):#10

    pilco.optimize_models()
    pilco.optimize_policy()

    X_new, Y_new, _, reward_ges = rollout(env=env, pilco=pilco, timesteps=50, render=True)
    # Update dataset
    REWARD.append(reward_ges)
    #print(REWARD)
    X = np.vstack((X, X_new));
    Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))
    #if rollouts == np.argmax(REWARD):
        #pilco.save_model()


#pilco.save_model()


#def load_model_if_exists(self):
   # model_path = os.path.abspath(os.getcwd()) + '/model.pth'
    #if os.path.exists(model_path):
        #controller_old = tf.saved_model.load(model_path)
        #self.controller.models = controller_old.models
        #self.controller.optimizers = controller_old.optimizers

#pilco.load_model()

import matplotlib.pyplot as plt

plt.figure()
plt.plot(REWARD)
plt.show()

print(REWARD)


