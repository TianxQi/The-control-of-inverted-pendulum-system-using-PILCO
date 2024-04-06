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

def save_data(X, Y, filename='data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((X, Y), f)

env = IPEnv()

X, Y, _, _ = rollout(env=env, pilco=None, random=True, timesteps=100, render=True)
for i in range(1, 5):  # 迭代 4 次
    X_, Y_, _, _ = rollout(env=env, pilco=None, random=True, timesteps=100, render=True)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

save_data(X, Y)