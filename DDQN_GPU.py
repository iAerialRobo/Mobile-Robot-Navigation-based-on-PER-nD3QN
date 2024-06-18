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


from environment_stage_2_28 import Env
from std_msgs.msg import Float32MultiArray
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 10
if torch.cuda.is_available:
    torch.cuda.manual_seed(seed)  
torch.manual_seed(seed)

dirPath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# tb=SummaryWriter("/home/wendy/Turtle_pytorch/src/dqn/tb")
# 超参数
BATCH_SIZE = 512                             # 样本数量
LR = 0.0005                                     # 学习率
EPISODES=0.99                          # greedy policy
GAMMA = 0.9                                    # reward discount
TARGET_REPLACE_ITER =200                      # 目标网络更新频率
MEMORY_CAPACITY = 200000                          # 记忆库容量
N_ACTIONS =5
env =Env(N_ACTIONS)        # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)            
N_STATES =28 # 杆子状态个数 (4个)
load_models=True

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

        self.fc1 = nn.Linear(N_STATES,256)                                # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 =nn.Linear(256,256)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3=nn.Linear(256,256)
        self.fc3.weight.data.normal_(0,0.1)
        self.out = nn.Linear(256,N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x=F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x=F.relu(self.fc2(x))
        x=F.dropout(self.fc2(x),p=0.2)
        x=F.relu(self.fc3(x))
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值

# # 定义Net类 (定义网络)
# class Net(nn.Module):
#     def __init__(self):                                                         # 定义Net的一系列属性
#         # nn.Module的子类函数必须在构造函数中执行父类的构造函数
#         super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

#         self.fc1 = nn.Linear(N_STATES, 250)                                # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
#         self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
#         self.fc2 =nn.Linear(250,250)
#         self.fc2.weight.data.normal_(0,0.1)
#         self.out = nn.Linear(250,N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
#         self.out.weight.data.normal_(0, 0.1)                                  # 权重初始化 (均值为0，方差为0.1的正态分布)

#     def forward(self, x):                                                       # 定义forward函数 (x为状态)
#         x=F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
#         x=F.relu(self.fc2(x))
#         # x=F.dropout(self.fc2(x))
#         x=F.dropout(self.fc2(x))
#         actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
#         return actions_value                                                    # 返回动作值


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):  # 定义DQN的一系列属性
        # self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.use_noisy=False
        self.grad_clip=10
        self.eval_net, self.target_net = Net().to(device), Net().to(device)                         # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 59))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        # self.loss_func = nn.MSELoss()      # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)     
        self.loss_func =  nn.MSELoss().to(device)  
        self.start_epoch = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay=0.99
        self.load_ep =780
        self.loss =0
        self.q_eval=0
        self.q_target=0
        self.use_soft_update= True
        self.tau=0.005
        self.max_q=0.0
        self.mean_eval=0.0
        if  load_models:
            checkpoint = torch.load("/home/wendy/RLprojects/ros_pytorch/src/dqn/src/Models/stage_2/perddqn/400a.pt")
            print(checkpoint.keys())
            print(checkpoint['epoch'])
            print(checkpoint)
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']
            self.epsilon=checkpoint["epsilon"]
            rospy.loginfo("loadmodel")
    def choose_action(self, x): # 定义动作选择函数 (x为状态)
        with torch.no_grad():
                x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)                           # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
                if np.random.uniform()>self.epsilon:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
                    actions_value = self.eval_net.forward(x)    # 通过对评估网络输入状态x，前向传播获得动作值
                    cpu_action_values = actions_value.cpu()
                    action = torch.max(cpu_action_values, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
                    action = action[0]                                             # 输出每一行最大值的索引，并转化为numpy ndarray形式                                               # 输出action的第一个数
                else:                                                                   # 随机选择动作
                    action = np.random.randint(0, N_ACTIONS) # 这里action随机等于0或1 (N_ACTIONS = 2)
                return action                                                           # 返回选择的动作 (0或1)


    def store_transition(self, s, a, r, s_,terminal):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_,terminal))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0 and (not self.use_soft_update):                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1 # 学习步数自加1
        with torch.autograd.set_detect_anomaly(True):
        # 抽取记忆库中的批数据
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
            b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
            b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
            # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
            b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
            # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
            b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
            # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
            b_s_ = torch.FloatTensor(b_memory[:, -(N_STATES+1):-1]).to(device)
            # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列
            b_terminals=torch.FloatTensor(b_memory[:, -1:]).to(device)
            ones = torch.ones(size=[BATCH_SIZE,1]).to(device)
            judge=ones-b_terminals
            # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
            q_eval = self.eval_net(b_s).gather(1, b_a)
            # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
            q_next = self.eval_net(b_s_).detach()
            # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
            next_actions = torch.argmax(q_next , dim=1, keepdim=True)
            target_q_values = self.target_net(b_s_).detach().gather(1, next_actions)
            # q_target  = self.target_net(next_state_batch).detach().max(1)[0].unsqueeze(1)
            q_target = b_r + GAMMA * judge*target_q_values
            # q_target = b_r + GAMMA *target_q_values
            # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
            loss = self.loss_func(q_eval, q_target)
            # 输入32个评估值和32个目标值，使用均方损失函数
            self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
            loss.backward()   # 误差反向传播, 计算参数更新值
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(dqn.eval_net.parameters(), self.grad_clip)
            self.optimizer.step() # 更新评估网络的所有参数
            self.max_q =torch.max(q_eval)
            self.mean_eval =torch.mean(q_eval)   
            if self.use_soft_update:  # soft update
                for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
       
         
        
    def save_model(self,dir,e,epsilon,rewards,total_sum_count,total_success_count,success_count_gap,sum_count_gap):
        state = {'target_net':self.target_net.state_dict(),'eval_net':self.eval_net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':e,'epsilon':epsilon,
                'rewards':rewards,'total_sum_count':total_sum_count,'total_success_count':total_success_count,'success_count_gap':success_count_gap,'sum_count_gap':sum_count_gap}
        torch.save(state,dirPath + '/Models/' + world + '/'+method+ '/'+str(e)+"a.pt")
    


method="ddqn"
world = 'stage_5'
if __name__=='__main__':
    stage_name="stage5"
    total_sum_count=0
    total_success_count=0
    steps_flag=0
    sum_count_episode=0
    success_count_episode=0
    success_count_gap=0
    sum_count_gap=0
    past_action = np.zeros(2)
    dqn = DQN()                                                             # 令dqn=DQN类
    rospy.init_node('DDQN_GPU')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=10)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=10)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time =time.time()
    e=dqn.start_epoch
    rewards=[]
    success_rates=[]
    writer = SummaryWriter("/home/wendy/RLprojects/ros_pytorch/src/dqn/src/logs/ddqn")
if  load_models:
        checkpoint = torch.load("/home/wendy/RLprojects/ros_pytorch/src/dqn/src/Models/stage_2/perddqn/400a.pt")
        rewards=checkpoint['rewards']
        total_sum_count=checkpoint['total_sum_count']
        total_success_count=checkpoint['total_success_count']
        success_count_gap=checkpoint['success_count_gap']
        sum_count_gap=checkpoint['sum_count_gap']
        rospy.loginfo("loadmodel")      
for e in range(1,5001):                                                    # 400个episode循环
    s = env.reset()                                                     # 重置环境                   s                        # 初始化该循环对应的episode的总奖励
    done=False
    episode_step=300
    rewards_current_episode = 0
    sum_count_episode=0
    success_count_episode=0
    for t in range(episode_step):                                                       # 开始一个episode (每一个循环代表一步)
        terminal=False
        a = dqn.choose_action(s)
        # s_, r, done,goal = env.step(a,t) 
        s_, r, done,goal = env.step(a)        
        if (done or goal):
            terminal= True    
            sum_count_episode=sum_count_episode+1
            total_sum_count=total_sum_count+1
            sum_count_gap=sum_count_gap+1
            if goal:
              success_count_episode=success_count_episode+1
              total_success_count=total_success_count+1
              success_count_gap=success_count_gap+1
                
        # if((t == episode_step-1) and (not done) and (not goal)):
        #     sum_count_episode=sum_count_episode+1
        #     total_sum_count=total_sum_count+1
        #     sum_count_gap=sum_count_gap+1
        dqn.store_transition(s, a, r, s_,terminal)                 # 存储样本
        rewards_current_episode = rewards_current_episode+r                           # 逐步加上一个episode内每个step的reward
        s = s_                                                # 更新状态  
        if dqn.memory_counter > BATCH_SIZE:              # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.learn()
        if e % 50 ==0:
            dqn.save_model(str(e),e,dqn.epsilon,rewards,total_sum_count,total_success_count,success_count_gap,sum_count_gap)
        if t >=episode_step-1:
            rospy.loginfo("time out!")
            # done =True
        if done or t == episode_step-1:  
            # success_rate=env.success_count/env.sum_count
            # success_rate_episode=success_count_episode/sum_count_episode
            total_success_rate=total_success_count/total_sum_count
            m,s =divmod(int(time.time()- start_time),60)
            h,m =divmod(m,60)
            rewards.append(rewards_current_episode)
            # success_rates.append(success_rate_episode)
            # writer.add_scalar("Max Q values:"+stage_name,dqn.max_q,e)
            # writer.add_scalar("Mean Q values:"+stage_name,dqn.mean_eval,e)
            writer.add_scalar("Total reward per episode:"+stage_name,rewards_current_episode,e)
            writer.add_scalar("Q values_Max"+stage_name,dqn.max_q,e)
            writer.add_scalar("Q values_Mean"+stage_name,dqn.mean_eval,e)
            rospy.loginfo('回合: %d 得分: %.2f 记忆量: %d 探索率: %.2f 花费时间: %d:%02d:%02d',e, rewards_current_episode, dqn.memory_counter, dqn.epsilon, h, m, s)
            # rospy.loginfo('sum count per 20 episodes:%d, success count per 20 episodes: %d,episode:%d',sum_count_gap,success_count_gap,e)
            # rospy.loginfo('total sum count :%d, total success count: %d,episode:%d',total_sum_count,total_success_count,e)
            # param_keys = ['epsilon']
            # param_values = [dqn.epsilon]
            # param_dictionary = dict(zip(param_keys, param_values))
            break
                                        
    if (not dqn.use_noisy) and (dqn.epsilon > dqn.epsilon_min):
            dqn.epsilon = dqn.epsilon *dqn.epsilon_decay
    # if dqn.epsilon > dqn.epsilon_min :
    #         dqn.epsilon =dqn.epsilon-0.0001
    average_reward = np.average(rewards)
    # average_success_rate=np.average(success_rates)
    writer.add_scalar("average Reward:"+stage_name,average_reward,e)
    # writer.add_scalar("success_rate_per_episode:"+stage_name,success_rate_episode,e)
    # writer.add_scalar("average succes rate:"+stage_name,average_success_rate,e) 
    writer.add_scalar("total succes rate:"+stage_name,total_success_rate,e) 
    if(e%20==0):
          average_success_rate_per_20_episodes=round(success_count_gap/sum_count_gap,4)
          writer.add_scalar("average success_rate per 20 episodes"+stage_name,average_success_rate_per_20_episodes,e) 
          sum_count_gap=0
          success_count_gap=0

writer.close()
        