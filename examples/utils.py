import numpy as np
from gpflow import config
from gym import make
float_type = config.default_float()


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False):
    X = []#state矩阵
    Y = []#每两次state之间的差#初始化两个空列表

    x = env.reset()#初始化环境状态
    ep_return_full = 0#采样奖励
    ep_return_sampled = 0#完整奖励
    for timestep in range(timesteps):#表示在模拟实验中的每个时间步
        if render: env.render()#如果它为 True，则调用 env.render()来在每个时间步渲染环境状态，通常用于可视化模拟过程。
        u = policy(env, pilco, x, random)#调用policy函数，根据当前状态x选择一个动作
        for i in range(SUBS):
            x_new, r, done, _ = env.step(u)#新的状态x_new、奖励r、done表示是否已完成的标志以及_其他信息
            x_1 = np.ravel(x_new)
            while x_1[2] > np.pi:
                x_1[2] -= 2 * np.pi
            while x_1[2] < -np.pi:
                x_1[2] += 2 * np.pi


            ep_return_full += r#前步骤的奖励r添加到ep_return_full中
            if done: break#退出子步骤的循环
            if render: env.render()#再次渲染环境状态，以显示子步骤的结果
        if verbose:#用于控制是否输出额外的信息，例如每个步骤的动作、状态和累积奖励。
            print("Action: ", u)

            print("State : ", x_1)#新状态#x_1
            print("Return so far: ", ep_return_full)#累积奖励
        X.append(np.hstack((x, u)))#当前时间步的状态x和动作u水平堆叠，然后添加到列表X中
        Y.append(x_1- x)#计算状态差
        ep_return_sampled += r#累积采样回合的奖励，将当前步骤的奖励r添加到ep_return_sampled中
        x = x_1
        #x = np.ravel(x_new)#更新当前状态x为新的状态x_new

        if done: break
    return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full

def rollout_with_switch(env, pilco_swing, pilco_stab, timesteps, verbose=True, random=False, SUBS=1, render=False):
    X = []  # state矩阵
    Y = []  # 每两次state之间的差#初始化两个空列表

    x = env.reset()  # 初始化环境状态
    ep_return_full = 0  # 采样奖励
    ep_return_sampled = 0  # 完整奖励
    for timestep in range(timesteps):  # 表示在模拟实验中的每个时间步
        if render: env.render()  # 如果它为 True，则调用 env.render()来在每个时间步渲染环境状态，通常用于可视化模拟过程。

        # 根据状态x中的theta值选择控制器
        theta = x[2]

        if -0.0176 < theta < 0.0176:
            pilco = pilco_stab
        else:
            pilco = pilco_swing

        u = policy(env, pilco, x, random)  # 调用policy函数，根据当前状态x和选定的pilco实例选择一个动作

        for i in range(SUBS):
            x_new, r, done, _ = env.step(u)  # 新的状态x_new、奖励r、done表示是否已完成的标志以及_其他信息
            x_1 = np.ravel(x_new)
            # 角度标准化移到循环外部进行
            while x_1[2] > np.pi:
                x_1[2] -= 2 * np.pi
            while x_1[2] < -np.pi:
                x_1[2] += 2 * np.pi

            ep_return_full += r  # 前步骤的奖励r添加到ep_return_full中
            if done: break  # 退出子步骤的循环
            if render: env.render()  # 再次渲染环境状态，以显示子步骤的结果
        if verbose:  # 用于控制是否输出额外的信息，例如每个步骤的动作、状态和累积奖励。
            print("Action: ", u)
            print("State : ", x_1)  # 新状态#x_1
            print("Return so far: ", ep_return_full)  # 累积奖励
        X.append(np.hstack((x, u)))  # 当前时间步的状态x和动作u水平堆叠，然后添加到列表X中
        Y.append(x_1 - x)  # 计算状态差#源代码x_1 = x_new
        ep_return_sampled += r  # 累积采样回合的奖励，将当前步骤的奖励r添加到ep_return_sampled中
        x = x_1  # 更新当前状态x为新的状态x_new

        if done: break
    return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full

def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]

class Normalised_Env():
    def __init__(self, env_id, m, std):
        self.env = make(env_id).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()
