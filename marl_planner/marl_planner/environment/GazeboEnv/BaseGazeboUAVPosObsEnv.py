import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
from marl_planner.environment.GazeboEnv.Quadrotor.utils.PubSubClasses import LidarSubscriber 
from marl_planner.environment.GazeboEnv.Quadrotor.BaseGazeboUAVPosEnv import BaseGazeboUAVPosEnv 

class BaseGazeboUAVPosObsEnv(BaseGazeboUAVPosEnv):
    
    def __init__(self): 
        
        super().__init__()

        self.lidar_subscriber = LidarSubscriber()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.uam_publisher)
        self.executor.add_node(self.lidar_subscriber)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.state_size = 363

    def step(self, action):
        
        action = action[0]
        self.q = self.q + action[:3]

        self.publish_simulator(self.q)

        lidar,self.check_contact = self.get_lidar_data()
        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward()
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:
            print(f"The position error at the end : {self.pose_error}")
            print(f"The constraint broken is : {self.const_broken}")
            print(f"The end pose of UAV is : {self.q[:3]}")

        pose_diff = self.q_des - self.q
        prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)

        if self.const_broken:
            self.q = self.q - action[:3]
            self.publish_simulator(self.q)

        self.current_time += 1

        return prp_state, reward, done, info

    def get_reward(self):
        
        done = False
        pose_error = self.pose_error

        if not self.const_broken:

            if pose_error < 0.1:
                done = True
                reward = 10
            elif pose_error < 0.5:
                reward = 5
            elif pose_error < 1:
                reward = 0
            else:
                reward = -(pose_error*10)
        
        else:
            reward = -20
        
        if self.current_time > self.max_time:
            done = True
            reward -= 2

        return reward,done
    
    def get_constraint(self):
        
        constraint = 0
        if self.const_broken:

            for i in range(self.q.shape[0]):
                if self.q[i] > self.max_q_bound[i]:
                    constraint+= (self.q[i] - self.max_q_bound[i])*10
                elif self.q[i] < self.min_q_bound[i]:
                    constraint+= abs(self.q[i] - self.min_q_bound[i])*10

            if constraint < 0:
                constraint = 10
        else:

            for i in range(self.q.shape[0]):
                constraint+= (abs(self.q[i]) - self.max_q_bound[i])*10

        return constraint

    def get_info(self,constraint):

        info = {}
        info["constraint"] = constraint
        info["safe_reward"] = -constraint
        info["safe_cost"] = 0
        info["negative_safe_cost"] = 0
        info["engage_reward"] = -10

        if np.any(self.q > self.max_q_safety) or np.any(self.q < self.min_q_safety):
            info["engage_reward"] = 10
            
        if constraint > 0:
            info["safe_cost"] = 1
            info["negative_safe_cost"] = -1

        return info

    def constraint_broken(self):
        

        if self.check_contact:
            return True
        
        if super().constraint_broken():
            return True
        
        return False
        
    def reset(self,pose = np.array([0,0,2]),pose_des = None,max_time = 10):

        prp_state = super().reset(pose,pose_des,max_time)
        lidar,self.check_contact = self.get_lidar_data()

        prp_state = np.concatenate((prp_state[0],lidar))
        prp_state = prp_state.reshape(1,-1)

        return prp_state

    def get_lidar_data(self):

        data,contact = self.lidar_subscriber.get_state()
        return data,contact  