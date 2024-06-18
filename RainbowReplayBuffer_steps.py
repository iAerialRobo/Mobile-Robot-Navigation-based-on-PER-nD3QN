
import heapq
import torch
import numpy as np
import rospy
from collections import deque

class SumTree():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size  # SumTree页节点数量=经验回放池容量
                                        # 储存Sum-Tree的所有节点数值
        self.tree = np.zeros(2*buffer_size-1) 
                                        # 储存经验数据，对应所有叶节点
        self.Transition = np.zeros(buffer_size, dtype=object)  
        self.TR_index = 0               # 经验数据的索引
        
    ## 向Sum-Tree中增加一个数据    
    def add(self, priority, expdata):   # priority优先级，expdata经验数据
                                        # TR_index在树中的位置为ST_index
        ST_index = self.TR_index+self.buffer_size-1
                                        # 将expdata存入TR_index位置
        self.Transition[self.TR_index] = expdata
                                        # 将TR_index的优先级priority存入
                                        # SumTree的ST_index位置，并更新SumTree
        
        self.update(ST_index, priority)  
        self.TR_index += 1              # 指针往前跳动一个位置
        if self.TR_index >= self.buffer_size:
            self.TR_index = 0           # 若容量已满，将叶节点指针拨回0
            
    ## 在ST_index位置添加priority后，更新Sum-Tree
    def update(self, ST_index, priority):
                                        # ST_index位置的优先级改变量
        change = priority-self.tree[ST_index] 
        self.tree[ST_index] = priority  # 将优先级存入叶节点
        while ST_index != 0:            # 回溯至根节点
            ST_index = (ST_index-1)//2  # 父节点 
            self.tree[ST_index] += change

    ## 根据value抽样
    def get_leaf(self, value):
        # rospy.loginfo('Search value:%.5f in SumTree,Sumtree TR_index is: %d',value,self.TR_index)
        parent_idx = 0                  # 父节点索引
        while True:     
            cl_idx = 2*parent_idx+1     # 左子节点索引
            cr_idx = cl_idx+1           # 右子节点索引        
            if cl_idx >= len(self.tree):# 检查是否已经遍历到底了 
                leaf_idx = parent_idx   # 父节点成为叶节点
                # rospy.loginfo('Search to %d in SumTree at last,leaf_idx is %d, related value is %.6f',cl_idx,leaf_idx,self.tree[leaf_idx])
                break                   # 已经到底了，停止遍历
            else:
                                        # value小于左子节点数值，遍历左子树
                if value <= self.tree[cl_idx]: 
                    parent_idx = cl_idx # 父节点更新，进入更下一层
                    # rospy.loginfo('Search to %d in SumTree, related value is %.6f',cl_idx,self.tree[cl_idx])
                    
                else:                   # 否则遍历右子树
                                        # 先减去左子节点数值
                    value -= self.tree[cl_idx] 
                    parent_idx = cr_idx # 父节点更新，进入更下一层
                    # rospy.loginfo('Search to %d in SumTree, related value is %.6f',cr_idx,self.tree[cr_idx])
                                        # 将Sum-tree索引转成Transition索引
        TR_index = leaf_idx-self.buffer_size+1 
        # if(self.tree[leaf_idx]==0):
        #    rospy.loginfo('priority is o,leaf_idx:%d',leaf_idx)
        
        return leaf_idx, self.tree[leaf_idx], self.Transition[TR_index]

    ## 根节点数值，即所有优先级总和
    def total_priority(self):
        return self.tree[0]
    def priority_max(self):
        
        max_priority = np.max(self.tree[-self.buffer_size:]) 
        return max_priority

class PrioritizedReplayMemory:
    def __init__(self, buffer_size,n_steps):
        self.tree = SumTree(buffer_size)    # 创建一个Sum-Tree实例
        self.counter = 0                    # 经验回放池中数据条数
        self.epsilon = 0.01                 # 正向偏移以避免优先级为0
        self.alpha = 0.6                    # [0,1],优先级使用程度系数
        self.beta = 0.4                     # 初始IS值
        self.delta_beta = 0.001             # beta增加的步长
        self.abs_err_upper = 64.  # TD误差绝对值的上界
        self.current_size=0
        self.buffer_size=buffer_size
        self.n_steps=n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.gamma=0.99
     
    ## 往经验回放池中装入一个新的经验数据
    def store(self, state, action, reward, next_state, terminal, done):    
        transition = (state, action, reward, next_state, terminal, done)
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            expdata = np.hstack((state, action, n_steps_reward, next_state,terminal))
            # 如果是buffer中的第一条经验，那么指定priority为1.0；否则对于新存入的经验，指定为当前最大的priority
            max_priority = 1.0 if self.current_size == 0 else self.tree.priority_max()
            self.tree.add(max_priority, expdata)
            self.counter = (self.counter+1)%self.buffer_size
            self.current_size = min(self.current_size + 1, self.buffer_size)
            
            
     
    def store_transition(self, state, action, reward, next_state, terminal, done):
        transition = (state, action, reward, next_state, terminal, done)
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            self.buffer['state'][self.count] = state
            self.buffer['action'][self.count] = action
            self.buffer['reward'][self.count] = n_steps_reward
            self.buffer['next_state'][self.count] = next_state
            self.buffer['terminal'][self.count] = terminal
            self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1, self.buffer_capacity)
       
            
  
    ## 从经验回放池中取出batch_size个数据s
    def sample(self, batch_size):
        zero_sample=0
        # indexes储存取出的优先级在SumTree中索引，一维向量
        # samples存储去除的经验数据，二维矩阵
        # ISWeights储存权重，以为向量
        indexes,samples,ISWeights = np.empty((batch_size,),dtype=np.int32),np.empty((batch_size,self.tree.Transition[0].size)),np.empty((batch_size,1))
        # 将优先级总和batch_size等分
        pri_seg = self.tree.total_priority()/batch_size 
        # IS值逐渐增加到1，然后保持不变
        self.beta = np.min([1., self.beta+self.delta_beta])  
        # 最小优先级占总优先级之比
        min_prob = np.min(self.tree.tree[-self.tree.buffer_size:])/self.tree.total_priority() 
        # 修正最小优先级占总优先级之比，当经验回放池未满和优先级为0时会用上
        if min_prob == 0: 
            min_prob = 0.00001
        for i in range(batch_size):
            while True:
                a,b = pri_seg*i,pri_seg*(i+1)   # 第i段优先级区间
                
                value = np.random.uniform(a,b)  # 在第i段优先级区间随机生成一个数
                # rospy.loginfo("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%random sample value:%d",value)
                # 返回SumTree中索引，优先级数值，对应的经验数据
                index,priority,sample = self.tree.get_leaf(value)
                                                # 抽样出的优先级占总优先级之比
                
                prob = priority/self.tree.total_priority()
                
                if(prob!=0):
                                                                             # 计算权重
                    ISWeights[i,0] = np.power(prob/min_prob,-self.beta)
                    indexes[i],samples[i,:] = index,sample
                    break
                else:
                    zero_sample+=1
                    rospy.loginfo('采样为0个数: %d',zero_sample)    
            
        return indexes, samples, ISWeights

    ## 调整批量数据
    def batch_update(self, ST_indexes, abs_errors):
        abs_errors += self.epsilon  # 加上一个正向偏移，避免为0
        
        priority=abs_errors**self.alpha
                                    # TD误差绝对值不要超过上界
        # clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        #                             # alpha决定在多大程度上使用优先级
        prioritys = np.power(abs_errors, self.alpha) 
        # rospy.loginfo('TD error related priority***********************************************************************')
        # rospy.loginfo(prioritys[:-1])

                                    # 更新优先级，同时更新树
        for index, priority in zip(ST_indexes, prioritys):
            self.tree.update(index, priority) 
        
        # rospy.loginfo('update sumTree after sampmle***********************************************************************')
        # rospy.loginfo(self.tree.tree[-self.tree.buffer_size:])
    def __len__(self):
        return len(self.tree)
   
    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0][:2]  # 获取deque中第一个transition的s和a
        next_state, terminal = self.n_steps_deque[-1][3:5]  # 获取deque中最后一个transition的s'和terminal
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):  # 逆序计算n_steps_reward
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if d:  # 如果done=True，说明一个回合结束，保存deque中当前这个transition的s'和terminal作为这个n_steps_transition的next_state和terminal
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal
    
    
    #   def get_n_steps_transition(self):
    #     state, action = self.n_steps_deque[0][:2]  # 获取deque中第一个transition的s和a
    #     next_state, terminal = self.n_steps_deque[-1][3:5]  # 获取deque中最后一个transition的s'和terminal
    #     n_steps_reward = 0
    #     for i in reversed(range(self.n_steps)):  # 逆序计算n_steps_reward
    #         r, s_, ter, d = self.n_steps_deque[i][2:]
    #         n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
    #         if d:  # 如果done=True，说明一个回合结束，保存deque中当前这个transition的s'和terminal作为这个n_steps_transition的next_state和terminal
    #             next_state, terminal = s_, ter

    #     return state, action, n_steps_reward, next_state, terminal

