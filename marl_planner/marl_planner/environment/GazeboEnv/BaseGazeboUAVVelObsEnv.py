import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
import time
from marl_planner.environment.GazeboEnv.Quadrotor.utils.PubSubClasses import LidarSubscriber 
from marl_planner.environment.GazeboEnv.Quadrotor.utils.CltSrvClasses import UavClientAsync, ResetSimClientAsync, GetUavPoseClientAsync, PauseGazeboClient, UnPauseGazeboClient
from marl_planner.environment.GazeboEnv.Quadrotor.BaseGazeboUAVVelEnv import BaseGazeboUAVVelEnv 

class BaseGazeboUAVVelObsEnv(BaseGazeboUAVVelEnv):
    
    def __init__(self): 

        super().__init__()
        
        self.lidar_subscriber = LidarSubscriber()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.uam_publisher)
        self.executor.add_node(self.get_uav_pose_client)
        self.executor.add_node(self.pause_sim)
        self.executor.add_node(self.unpause_sim)
        self.executor.add_node(self.lidar_subscriber)
        self.executor.add_node(self.reset_sim)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.state_size = 363
    
    def step(self, action):
        
        action = action[0]
        self.vel = self.vel + action[:3]

        self.vel = np.clip(self.vel,self.min_q_bound,self.max_q_bound)

        self.publish_simulator(self.vel)

        lidar,self.check_contact = self.get_lidar_data()
        self.get_uav_pose()
        # self.check_contact = self.collision_sub.get_collision_info()

        # print(f"New pose : {new_q}")
        # print(f"New velocity : {new_q_vel}")
        # self.q,self.qdot = self.controller.solve(new_q,new_q_vel)

        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward()
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:
            print(f"The constraint is broken : {self.const_broken}")
            print(f"The position error at the end : {self.pose_error}")
            print(f"The end pose of UAV is : {self.pose[:3]}")

        pose_diff = self.q_des - self.pose
        prp_state = np.concatenate((pose_diff,lidar))
        prp_state = prp_state.reshape(1,-1)
        self.current_time += 1

        if self.const_broken:

            self.get_safe_pose()
            uav_pos_ort = list(self.previous_pose)[0:3]
            uav_pos_ort += [0,0,0]
            uav_pos_ort = f"{uav_pos_ort}"[1:-1]

            self.reset_sim.send_request(uav_pos_ort)

            self.vel = self.vel - action[:3]
            # self.publish_simulator(self.vel)

        return prp_state, reward, done, info

    def get_reward(self):
        
        done = False
        pose_error = self.pose_error

        if not self.const_broken:
            self.previous_pose = self.pose
            # if pose_error < 0.01:
            #     done = True
            #     reward = 1000
            # elif pose_error < 0.05:
            #     done = True
            #     reward = 100
            if pose_error < 0.1:
                done = True
                reward = 10
            # elif pose_error < 1:
            #     done = True
            #     reward = 0
            else:
                reward = -(pose_error*5)
        
        else:
            reward = -20
            # done = True

        if self.current_time > self.max_time:
            done = True
            reward -= 2

        return reward,done
    
    def get_constraint(self):
        
        constraint = 0
        if self.const_broken:

            for i in range(self.vel.shape[0]):
                if self.vel[i] > self.max_q_bound[i]:
                    constraint+= (self.vel[i] - self.max_q_bound[i])*10
                elif self.vel[i] < self.min_q_bound[i]:
                    constraint+= abs(self.vel[i] - self.min_q_bound[i])*10

            if constraint < 0:
                constraint = 10
        else:

            for i in range(self.vel.shape[0]):
                constraint+= (abs(self.vel[i]) - self.max_q_bound[i])*10

        return constraint

    def get_info(self,constraint):

        info = {}
        info["constraint"] = constraint
        info["safe_reward"] = -constraint
        info["safe_cost"] = 0
        info["negative_safe_cost"] = 0
        info["engage_reward"] = -10

        if np.any(self.vel > self.max_q_safety) or np.any(self.vel < self.min_q_safety):
            info["engage_reward"] = 10
            
        if constraint > 0:
            info["safe_cost"] = 1
            info["negative_safe_cost"] = -1

        return info

    def constraint_broken(self):
        
        if self.check_contact:
            return True
        
        # if np.any(self.vel[:3] > self.max_q_bound[:3]) or np.any(self.vel[:3] < self.min_q_bound[:3]):
        #     return True
        
        return False
        
    def reset(self,pose = np.array([0,0,2]),pose_des = None,max_time = 10):
        
        prp_state = super().reset(pose,pose_des,max_time)
        lidar,self.check_contact = self.get_lidar_data()

        # pose_diff = np.clip(pose_diff,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((prp_state[0],lidar))
        prp_state = prp_state.reshape(1,-1)

        return prp_state
        
    def get_lidar_data(self):

        data,contact = self.lidar_subscriber.get_state()
        return data,contact

    def get_safe_pose(self):

        for i in range(len(self.previous_pose) - 1):

            if self.previous_pose[i] < 0:
                self.previous_pose[i]+= 0.05
            else:
                self.previous_pose[i]-= 0.05

if __name__ == "__main__":

    rclpy.init()

    env = BaseGazeboUAVVelObsEnv()
    env.reset()

    action = np.array([0,0,0,0,0,0,0]).reshape(1,-1)

    prp_state, reward, done, info = env.step(action)

    print(env.q)
    print(info["constraint"])