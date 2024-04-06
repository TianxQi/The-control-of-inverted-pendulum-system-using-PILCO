import gpflow.config
import numpy as np
from pilco.models import PILCO
from pilco.controllers import RbfController
import pickle
from utils import policy
from utils import rollout,rollout_with_switch
import gpflow
from examples.IPenv import IPEnv
import tensorflow as tf
float_type = gpflow.config.default_float()
env= IPEnv()
def load_model(filename='best_model.pkl'):
    with open(filename, 'rb') as f:
        model_data_swing = pickle.load(f)
    return model_data_swing
def load_data(filename='data.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)
# def load_model(filename='best_model_try.pkl'):
#     with open(filename, 'rb') as f:
#         model_data_stab = pickle.load(f)
#     return model_data_stab
# # 使用上述函数加载模型
model_data_swing = load_model('best_model.pkl')
model_data_stab = load_model('best_model_try.pkl')
# 从加载的字典中提取 controller
controller_swing =model_data_swing['controller']#[2.66060032,0.74840393,5.91644645 ,10.58763371]# model_data_swing['controller']
#controller_swing_matrix = np.array(controller_swing).reshape(4, 1)
controller_stab =model_data_stab['controller']#[[1.79989429],[2.41270613],[2.00522412],[2.49754763]]# model_data_stab['controller']
#controller_stab_matrix = np.array(controller_swing).reshape(4, 1)
state = np.array([0,0,np.pi,0])

gravity = 9.81  # m/s**2
masscart = 0.5862  # kg
masspole = 0.1342  # kg
total_mass = masscart + masspole  # kg
length = 0.49  # m
interia = (masspole * (length ** 2)) / 12
tau = 0.05  # 0.01
T_1 = 0.0395

x = state.item(0)
x_dot = state.item(1)
theta = state.item(2)
theta_dot = state.item(3)
X_swing,Y_swing = load_data('data_swing.pkl')
X_stab,Y_stab = load_data('data_stab.pkl')

pilco_swing = PILCO((X_swing, Y_swing), controller=controller_swing, horizon=40)
pilco_stab = PILCO((X_stab, Y_stab), controller=controller_stab, horizon=40)
def func(state, u):
    x = state.item(0)  # 从输入的 state 中提取出来变量
    x_dot = state.item(1)
    theta = state.item(2)
    theta_dot = state.item(3)

    x_dot2 = -1 / T_1 * x_dot + 1 / T_1 * u
    theta_dot2 = 1.5 / length * (gravity * np.sin(theta) - x_dot2 * np.cos(theta))

    Q_dot = np.array([[x_dot], [x_dot2], [theta_dot], [theta_dot2]])
    return Q_dot



s = tf.zeros((4, 4))
max_step = 100
X_new, Y_new, _, reward_ges=rollout_with_switch(env, pilco_swing, pilco_stab, timesteps=max_step, verbose=True, random=False, SUBS=1, render=True )

#X_new, Y_new, _, reward_ges = rollout(env=env, pilco=pilco_swing, timesteps=100, render=True)