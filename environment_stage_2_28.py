#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from respawnGoal import Respawn
from respawnGoal_official import Respawn
from torch.utils.tensorboard import SummaryWriter
STEPS_EVELUATE=50
# stage_name="4"

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.sum_count=0
        self.success_count=0
        self.success_rate=0.0  
        self.past_distance=0
        self.success_steps_rate=0.0
        # self.past_u=0
        # self.thresh_obstacle=0.5
        # self.thresh_goal=0.5
        # self.sigma=100
        # self.miu=100
        
    def getGoalDistace(self):
        goalPositionCheck = True        #Goal position is invalid
        while(goalPositionCheck):
             goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
             if(goal_distance==0):
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)  
             else:
                goalPositionCheck = False
        self.past_distance = goal_distance
        return goal_distance
    # def getGoalDistace(self):
    #     goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
    #    return goal_distance
    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True
        

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done,self.get_goalbox

    def setReward(self, state, done, action):
        yaw_reward = []
        goal_reward=0
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        if obstacle_min_range < 0.5:
            ob_reward = -5
        else:
            ob_reward =1

        # reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward+goal_reward
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward
        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())   #when collision, this episode end 
            self.sum_count=self.sum_count+1      
        if self.get_goalbox:                 #when get goal, generate new goal to serach untial collision or reach max steps in each episode.
            rospy.loginfo("Goal!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)  
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
            self.sum_count=self.sum_count+1
            self.success_count=self.success_count+1
        return reward
    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done, goal= self.getState(data)
        reward = self.setReward(state, done, action)
        # if self.sum_count!=0:
        #     self.success_rate=self.success_count*1.0/self.sum_count
        return np.asarray(state), reward, done,goal

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done,goal= self.getState(data)
        
        return np.asarray(state)