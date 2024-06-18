#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
# import gym                                      # 导入gym
import math, random
import os
import sys
import copy
import time
import rospy


from environment_stage_2 import Env
from std_msgs.msg import Float32MultiArray
from torch.utils.tensorboard import SummaryWriter
from preMemory_new import ReplayMemory_Per,Transition

dirPath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# tb=SummaryWriter("/home/wendy/Turtle_pytorch/src/dqn/tb")
# 超参数
BATCH_SIZE = 512                              # 样本数量
LR = 0.001                                       # 学习率
EPISODES=0.99                          # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 10                       # 目标网络更新频率
MEMORY_CAPACITY = 200000                          # 记忆库容量
N_ACTIONS = 5
env =Env(N_ACTIONS)        # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)            
N_STATES =  26  # 杆子状态个数 (4个)


"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。
定义网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中。
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
"""


# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 64)                                # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 =nn.Linear(64,64)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(64,N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x=F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x=F.relu(self.fc2(x))
        # x=F.dropout(self.fc2(x))
        x=F.dropout(self.fc2(x))
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):  # 定义DQN的一系列属性
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = ReplayMemory_Per(MEMORY_CAPACITY)           # 初始化记忆库，一行代表一个transition       
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        # self.loss_func = nn.MSELoss()      # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)     
        self.loss_func =  nn.MSELoss()  
        self.start_epoch = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.load_models=False
        self.load_ep =780
        self.loss =0
        self.q_eval=0
        self.q_target=0
        if self.load_models:
            self.epsilon= 0
            self.start_epoch=self.load_ep
            checkpoint = torch.load("/home/robot/catkin_ws/src/dqn/src/Models/stage_2/290a.pt")
            print(checkpoint.keys())
            print(checkpoint['epoch'])
            print(checkpoint)
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            rospy.loginfo("loadmodel")
    def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() > self.epsilon:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式

            action = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action                                                           # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
 
        self.memory.push(transition)
                                 
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1
        
        states, actions, rewards, next_states, dones, is_weights, batch = self.memory.sample(BATCH_SIZE)
        # b_s = torch.FloatTensor(batch[:, :N_STATES])
        # # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        # b_a = torch.LongTensor(batch[:, N_STATES:N_STATES+1].astype(int))
        # # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        # b_r = torch.FloatTensor(batch[:, N_STATES+1:N_STATES+2])
        # # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        # b_s_ = torch.FloatTensor(batch[:, -N_STATES:])
        # # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(states).gather(1, actions)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.eval_net(next_states).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        next_actions = torch.argmax(q_next , dim=1, keepdim=True)
        target_q_values = self.target_net(next_states).detach().gather(1, next_actions)
        # q_target  = self.target_net(next_state_batch).detach().max(1)[0].unsqueeze(1)
        q_target = rewards+ GAMMA * target_q_values
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        self.loss = torch.max(loss)
        self.q_eval =torch.max(q_eval)
        self.q_target =torch.max(q_target)
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                          # 更新评估网络的所有参数



        

    def save_model(self,dir,e):
        state = {'target_net':self.target_net.state_dict(),'eval_net':self.eval_net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':e}
        torch.save(state,dirPath + '/Models/' + world + '/'+str(e)+"a.pt")
    



world = 'stage_2'
if __name__=='__main__':
    past_action = np.zeros(2)
    dqn = DQN()                                                             # 令dqn=DQN类
    rospy.init_node('DQN2')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=10)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=10)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time =time.time()
    e=dqn.start_epoch
    
    
for e in range(300):                                                    # 400个episode循环
    s = env.reset()                                                     # 重置环境                                            # 初始化该循环对应的episode的总奖励
    done=False
    episode_step=500
    rewards_current_episode = 0
    for t in range(episode_step):                                                       # 开始一个episode (每一个循环代表一步)
        a = dqn.choose_action(s)
        s_, r, done = env.step(a)     
        # 输入该步对应的状态s，选择动作
        # s_, r, done, success_rate = env.step(a)          # 执行动作，获得反馈
        # # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # new_r = r1 + r2
        dqn.store_transition(s, a, r, s_)                 # 存储样本
        rewards_current_episode = rewards_current_episode+r                           # 逐步加上一个episode内每个step的reward

        s = s_                                                # 更新状态
        pub_get_action.publish(get_action)   
        
        if dqn.memory_counter > BATCH_SIZE:              # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.learn()
        if e % 10 ==0:
            dqn.save_model(str(e),e)
        if t >=500:
            rospy.loginfo("time out!")
            done =True


        if done or t == episode_step-1:     # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            m,s =divmod(int(time.time()- start_time),60)
            h,m =divmod(m,60)
            rospy.loginfo('回合: %d 得分: %.2f 记忆量: %d 探索率: %.2f 花费时间: %d:%02d:%02d',e, rewards_current_episode, dqn.memory_counter, dqn.epsilon, h, m, s)
   
            param_keys = ['epsilon']
            param_values = [dqn.epsilon]
            param_dictionary = dict(zip(param_keys, param_values))
            # AverageReward = rewards_current_episode*1.0/t
            result.data = [rewards_current_episode, dqn.q_eval]
            pub_result.publish(result)
           
            print('reward per ep: ' + str(rewards_current_episode))
            print('*\nbreak step: ' + str(t) + '\n*')
            break
                                        
        if dqn.epsilon > dqn.epsilon_min :
            dqn.epsilon =dqn.epsilon-0.0001
