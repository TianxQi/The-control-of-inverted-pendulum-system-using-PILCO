import gym
import numpy as np

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from gym import spaces
#def normalize_angle(angle):#将角度 规范化为区间 [-π, π] 内的值
    ###3from the closest multiple of 2*pi)
    #"""
    #while angle > np.pi:
          #angle -= 2 * np.pi
    #while angle < -np.pi:
          #angle += 2 * np.pi
    #return angle

class IPEnv_SAC(gym.Env):

   # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.gravity = 9.81#m/s**2
        self.masscart = 0.5862#kg
        self.masspole = 0.1342#kg
        self.total_mass = self.masscart + self.masspole#kg
        self.length =0.49#m
        self.interia = (self.masspole * (self.length** 2))/12
        self.tau = 0.005#0.01
        self.T_1 = 0.0395

     #self.kinematics_integrator = "euler"#欧拉积分法
        self.viewer = None#没有创建用于可视化的窗口
        self.scale = 100  # 每个单位长度将等于100像素
        self.seed=np.random.seed()#随机数种子
        self.state = None  # 还没有定义环境的初始状态
        self.steps_beyond_done = None#用于跟踪在环境中超出了任务完成状态的步数。还没有超出任务完成状态的步数

     #.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 0.44#小车位置的阈值
        self.x_min = -(self.x_threshold )
        self.x_max = (self.x_threshold )
        self.theta_min = -np.pi#摆杆角度的最小值和最大值。在这里被设置为浮点数类型的最小负值和最大值，表示没有限制摆杆角度的范围。
        self.theta_max = np.pi
        self.velocity_min = -4#速度的最小值和最大值
        self.velocity_max = 4
        self.theta_dot_min = -10
        self.theta_dot_max = 10
     #定义了观测空间的最小值和最大值。观测空间是一个包含系统状态的空间，包括速度、角度和小车位置
        self.min_observation = np.array([self.x_min,
                                            self.velocity_min,
                                            self.theta_min,
                                            self.theta_dot_min],
                                     dtype=np.float32)

        self.max_observation = np.array([self.x_max,
                                             self.velocity_max,
                                             self.theta_max,
                                             self.theta_dot_max],
                                     dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_observation,
                                         high=self.max_observation,
                                         dtype=np.float32)
        self.min_action = -1 # min cart reference velocity
        self.max_action = 1 # max cart reference velocity

        self.action_space = spaces.Box(low=self.min_action,
                                    high=self.max_action,
                                    shape=(1,),
                                    dtype=np.float32)
     #self.action_space = spaces.Discrete(2)  # 用0和1分别表示向左或者向右的力
     #self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.x_goal_position = 0

        self.counter = 0  # Taken from fregu856

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #state = self.state
        state = np.ravel(self.state)

        x = state.item(0)
        x_dot = state.item(1)
        theta = state.item(2)
        #theta = normalize_angle(state.item(2))
        theta_dot = state.item(3)

        #state = np.array([x, x_dot, theta, theta_dot])#state被重新组织成一个4x1的矩阵

        u = action[0]
        self.counter += 1#递增计数器counter，用于跟踪环境中的步数

        # print("Count: ", self.counter)

        def func(state, u):
            x = state.item(0)#从输入的 state 中提取出来变量
            x_dot = state.item(1)
            theta = state.item(2)
            theta_dot = state.item(3)


            # print("Stateshape_initial:", state.shape)
            x_dot2 = -1/self.T_1 * x_dot +1/self.T_1 *u
            theta_dot2 = 1.5/self.length *(self.gravity * np.sin(theta) - x_dot2 * np.cos(theta))
            #A_Q=np.matrix([[0,1,0,0],[0,-1/self.T_1,0,0],[0,0,0,1],[0,1.5/(self.T_1*self.length),0,0]])
            #B_Q=np.matrix([[0],[1/self.T_1],[0],[-1.5/(self.T_1*self.length)]])
            #Q_dot = A_Q*state +B_Q * u
            Q_dot = [[x_dot],[x_dot2],[theta_dot],[theta_dot2]]
            return Q_dot

        state_dot = func(state, u)
        #state_dot = np.ravel(func(state, u))
        state_dot_new = np.matrix(state_dot)
        #self.state = self.state + self.tau * state_dot
        self.state = (np.reshape(self.state, (4, 1)) + self.tau * state_dot_new)#np.array([state_dot[0],state_dot[1],state_dot[2],state_dot[3]])self.state 被重塑为一个4x1的矩阵，确保它的维度与 state_dot_new 一致
        self.state = np.ravel(self.state)##使用 np.squeeze 函数来消除状态数组中的冗余维度。（4,1）-->(4,) 的一维数组
        #self.state[2]=normalize_angle(self.state[2])



        theta = self.state[2]
        #theta1 = theta
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        #theta2 = theta
        alive_bonus = 0
        x_tip = x + self.length * np.sin(theta) #摆杆的水平末端位置
        y_tip = self.length * np.cos(theta) #摆杆的垂直末端位置
        dist_penalty = ( x_tip  ** 2 +(y_tip - self.length) ** 2)#距离惩罚项
        velocity_penalty =(0.001 * (theta_dot ** 2)) + (0.003 * (x_dot ** 2))#速度惩罚项np.abs(x_dot)/2# (0.001 * (theta_dot ** 2)) + (0.005 * (x_dot ** 2))#速度惩罚项
        #angel_penatly = np.cos(theta)#(theta/np.pi)**2
        #angel_reward = 1-(theta/np.pi)**2
        #d = np.abs(theta_dot+5.39*theta)
        angel_reward2 =np.exp(-np.abs(theta/0.5))
        #angel_reward2 = 1.5 * np.exp(-np.abs(theta)) - 0.25
        #angel_reward3= -np.abs(theta)+3
        if -0.45<theta<0.45:#-0.05<theta<0.05:
            theta_dot_reward = np.exp(-np.abs(theta/0.5))
            #theta_dot_reward =-0.1*d**2
            #theta_dot_reward = -d/3
            #theta_dot_reward = np.exp(-d)-0.5
        else:
            theta_dot_reward = 0
        reward =0 -dist_penalty+angel_reward2-velocity_penalty#postion_reward - velocity_penalty
        done = bool(
            self.counter > 1600 or x < -self.x_threshold or x > self.x_threshold)  # or theta > 90*2*np.pi/360  or theta < -90*2*np.pi/360)
        if done:
            print("Self.counter: ", self.counter)

        if -0.1 < x < 0.1 and -0.1 < x_dot < 0.1 and -0.05 < theta_dot < 0.05 and -0.05 < theta < 0.05:  # 获得额外奖励
            reward += 1500#95.0
        #theta_target = 0
        if  -0.15 < x_dot < 0.15  and -0.12 < theta < 0.12:  # 获得额外奖励
            reward += 1.0

        if x < -self.x_threshold or x > self.x_threshold:#扣分
            reward -= 1500#95.0
        return self.state, reward, done, {}


    def reset(self):
    # Reset the state
        self.state = np.array([
           np.random.uniform(low=0, high=0),  # x
           np.random.uniform(low=0, high=0),  # x_dot
           np.random.uniform(low=np.pi, high=np.pi),  # theta np.pi
           np.random.uniform(low=0, high=0) # theta_dot
        ])

        self.steps_beyond_done = None
        self.counter = 0
        return self.state


    def render(self, return_image=True):
        plt.cla()#清空当前的图形，以便在每次渲染时绘制新的图像
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)#设置了图形的 x 和 y 轴的范围
       # plt.gca().set_aspect('equal')  # Set aspect ratio to equal

        cart_width = 0.02
        cart_height = 0.02#小车的宽度和高度
        self.state=np.ravel(self.state)
    # print("self.stateMatrixShape:", self.state.shape)
        cart_x = self.state[0] #从状态向量 self.state 中获取 state0=x
        cart_y = float(0) #小车的当前位置
        pendulum_x = cart_x + self.length * np.sin(self.state[2])##从状态向量 self.state中获取state2=theta
        pendulum_y = cart_y + self.length * np.cos(self.state[2])
    #摆杆的末端位置。


        cart_x_left = cart_x - cart_width / 2
        cart_x_right = cart_x + cart_width / 2#小车左侧和右侧的 x 坐标
        cart_y_top = cart_y + cart_height / 2
        cart_y_bottom = cart_y - cart_height / 2#小车顶部和底部的 y 坐标

        rect_vertices = np.array([[cart_x_left, cart_y_bottom],
                                 [cart_x_right, cart_y_bottom],
                                 [cart_x_right, cart_y_top],
                                 [cart_x_left, cart_y_top],
                                 [cart_x_left, cart_y_bottom]])
    #小车四个角坐标的 NumPy 数组，用于绘制小车的矩形
    # Plot the rectangular patch
        plt.plot(rect_vertices[:, 0], rect_vertices[:, 1], 'k-')  # Connect vertices to form rectangle用于绘制小车的矩形，连接矩形的四个角点以形成矩形轮廓
        plt.plot(cart_x, cart_y, 'ko',markersize=10)  # Plot a marker at the center of the rectangle        plt.gca().add_patch(cart_rect)
                             #绘制小车的中心点，以黑色圆点表示
        plt.plot([cart_x, pendulum_x], [cart_y, pendulum_y], 'r-')
    #绘制摆杆1的末端点，以红色圆点表示。
    # plt.plot(cart_x, cart_y, 'ko', markersize=10)
        plt.plot(pendulum_x, pendulum_y, 'ro', markersize=5)


        plt.pause(0.001)
        plt.draw()#用于暂停图形的绘制，然后重新绘制图形，以实现动态效果

        if return_image:#若return_image 参数为 True
           buf = BytesIO()
           plt.savefig(buf, format='png')#将绘制的图像以 PNG 格式保存到内存缓冲区中，并返回图像对象
           buf.seek(0)
           img = Image.open(buf)
           return img

    def close(self):
        plt.close()

