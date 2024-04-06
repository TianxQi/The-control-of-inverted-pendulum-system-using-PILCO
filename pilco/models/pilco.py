import numpy as np
import tensorflow as tf
import gpflow
import pandas as pd
import time
import os
from .mgpr import MGPR
from .smgpr import SMGPR
from .. import controllers
from .. import rewards

float_type = gpflow.config.default_float()
from gpflow import set_trainable

class PILCO(gpflow.models.BayesianModel):
    def __init__(self, data, num_induced_points=None, horizon=30, controller=None,
                reward=None, m_init=None, S_init=None, name=None):
        super(PILCO, self).__init__(name)
        if num_induced_points is None:
            self.mgpr = MGPR(data)
        else:
            self.mgpr = SMGPR(data, num_induced_points) # 根据提供的数据创建一个高斯过程回归模型
        self.state_dim = data[1].shape[1]
        self.control_dim = data[0].shape[1] - data[1].shape[1]
        self.horizon = horizon

        if controller is None:
            self.controller = controllers.LinearController(self.state_dim, self.control_dim)
        else:
            self.controller = controller

        if reward is None:
            self.reward = rewards.ExponentialReward(self.state_dim)
        else:
            self.reward = reward

        if m_init is None or S_init is None:
            # If the user has not provided an initial state for the rollouts,
            # then define it as the first state in the dataset.
            self.m_init = data[0][0:1, 0:self.state_dim]
            self.S_init = np.diag(np.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.S_init = S_init
        self.optimizer = None
    # mgpr代表高斯过程回归模型;state_dim和control_dim分别表示状态和控制的维度;m_init和S_init分别代表初始状态的均值和协方差
    def training_loss(self):
        # This is for tuning controller's parameters
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return -reward

    def optimize_models(self, maxiter=200, restarts=1):
        '''
        Optimize GP models
        '''
        self.mgpr.optimize(restarts=restarts) # 调用 mgpr 对象（多元高斯过程回归模型）的optimize方法来优化模型 restarts 参数用于控制优化过程的重启次数。
        # Print the resulting model parameters
        # ToDo: only do this if verbosity is large enough
        lengthscales = {}; variances = {}; noises = {}; # 核长度，方差和噪声
        i = 0
        for model in self.mgpr.models:
            lengthscales['GP' + str(i)] = model.kernel.lengthscales.numpy()
            variances['GP' + str(i)] = np.array([model.kernel.variance.numpy()])
            noises['GP' + str(i)] = np.array([model.likelihood.variance.numpy()]) # 获取这些参数的数值
            i += 1
        print('-----Learned models------')
        pd.set_option('display.precision', 3) # Pandas 显示精度
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises)) # 分别打印长度尺度（lengthscales）方差（variances）和噪声（noises）的数据框

    def optimize_policy(self, maxiter=50, restarts=1): # 指定优化过程的最大迭代次数50 优化过程的重新启动次数为1，用于寻找全局最优解
        '''
        Optimize controller's parameter's
        '''
        start = time.time() # 记录优化开始的时间
        mgpr_trainable_params = self.mgpr.trainable_parameters # 获取高斯过程模型的可训练参数
        for param in mgpr_trainable_params:
            set_trainable(param, False)

        if not self.optimizer:
            self.optimizer = gpflow.optimizers.Scipy() # 创建一个新的 gpflow.optimizers.Scipy 优化器
            # self.optimizer = tf.optimizers.Adam()
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter)) # 优化器的minimize方法优化training_loss函数以寻找最佳控制策略参数self.trainable_variables包含控制器的可训练参数
            # self.optimizer.minimize(self.training_loss, self.trainable_variables)
        else:
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter))
            # self.optimizer.minimize(self.training_loss, self.trainable_variables)
        end = time.time()
        print("Controller's optimization: done in %.1f seconds with reward=%.3f." % (end - start, self.compute_reward()))
        restarts -= 1
         # 在优化控制策略时进行重启，以寻找更优的参数设置 用于避免在局部最优解中陷入，从而有可能找到全局最优解
        best_parameter_values = [param.numpy() for param in self.trainable_parameters] # 初始化最佳参数为当前的可训练参数值
        best_reward = self.compute_reward() # 调用compute_reward函数计算当前控制策略的奖励，并将其设置为最佳奖励
        for restart in range(restarts):
            self.controller.randomize() # 随机化控制器的参数，以从不同的初始点开始优化
            start = time.time() # 记录优化开始的时间
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter))
            end = time.time() # 记录优化结束的时间
            reward = self.compute_reward() # 计算优化后控制策略的奖励
            print("Controller's optimization: done in %.1f seconds with reward=%.3f." % (end - start, self.compute_reward()))
            if reward > best_reward: # 新的奖励比之前记录的最佳奖励高，则更新最佳参数和奖励
                best_parameter_values = [param.numpy() for param in self.trainable_parameters] # 更新最佳参数为当前的参数值
                best_reward = reward # 更新最佳奖励

        for i,param in enumerate(self.trainable_parameters): # 遍历所有可训练参数
            param.assign(best_parameter_values[i]) # 将每个参数更新为找到的最佳参数值
        end = time.time()
        for param in mgpr_trainable_params:
            set_trainable(param, True) # 优化完成后将高斯过程模型的参数重新设置为可训练状态

    def compute_action(self, x_m):
        return self.controller.compute_action(x_m, tf.zeros([self.state_dim, self.state_dim], float_type))[0]

    def predict(self, m_x, s_x, n):
        loop_vars = [
            tf.constant(0, tf.int32),
            m_x,
            s_x,
            tf.constant([[0]], float_type)
        ]

        _, m_x, s_x, reward = tf.while_loop(
            # Termination condition
            lambda j, m_x, s_x, reward: j < n,
            # Body function
            lambda j, m_x, s_x, reward: (
                j + 1,
                *self.propagate(m_x, s_x),
                tf.add(reward, self.reward.compute_reward(m_x, s_x)[0])
            ), loop_vars
        )
        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x@c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x
        #TODO: cleanup the following line
        S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.state_dim]); S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x

    def compute_reward(self):
        return -self.training_loss()

    @property
    def maximum_log_likelihood_objective(self):
        return -self.training_loss()
    #def load_model_if_exists(self):
        #model_path = os.path.abspath(os.getcwd()) + '/model.pth'
        #if os.path.exists(model_path):
            #controller_old = tf.saved_model.load(model_path)
            #self.controller.models = controller_old.models
            #self.controller.optimizers = controller_old.optimizers
    #def save_model(self):
        #path = os.path.abspath(os.getcwd())
        #self.save_path = path +'/model.pth'
        #tf.saved_model.save(self.controller,self.save_path)
    

    # def load_model(self):
    # restore = tf.saved_model.load(self.save_path)

    # self.controller =  restore.signatures['__saved_model_init_op']
    # return restore
