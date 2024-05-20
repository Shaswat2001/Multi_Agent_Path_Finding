import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
from marl_planner.environment.GazeboEnv.Quadrotor.utils.CltSrvClasses import UavClientAsync

class BaseGazeboUAVPosEnv(gym.Env):
    
    def __init__(self): 
        

        self.uam_publisher = UavClientAsync()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.uam_publisher)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.state = None
        self.state_size = 3
        self.action_max = np.array([0.1,0.1,0.1])
        
        self.q = None
        self.q_des = None
        
        self.max_time = 10
        self.dt = 0.01
        self.current_time = 0

        self.q_vel_bound = np.array([3,3,3,1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        self.max_q_bound = np.array([12,12,23,2*np.pi,0.4,3,1.3])
        self.min_q_bound = np.array([-12,-12,0.6,-2*np.pi,-2.7,-0.2,-3.4])

        self.max_q_safety = np.array([8,8,8])
        self.min_q_safety = np.array([-8,-8,2])
        # self.max_q_safety = None
        # self.min_q_safety = None

        self.max_safety_engage = np.array([5.5,5.5,5.5])
        self.min_safety_engage = np.array([-5.5,-5.5,0.8])

        self.safe_action_max = np.array([8,8,8])
        self.safe_action_min = np.array([-8,-8,2])
        self.check_contact = False

        self.action_space = spaces.Box(-self.action_max,self.action_max,dtype=np.float64)

    def step(self, action):
        
        action = action[0]
        self.q = self.q + action[:3]

        self.publish_simulator(self.q)

        # print(f"New pose : {new_q}")
        # print(f"New velocity : {new_q_vel}")
        # self.q,self.qdot = self.controller.solve(new_q,new_q_vel)

        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward()
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:
            print(f"The position error at the end : {self.pose_error}")
            print(f"The end pose of UAV is : {self.q[:3]}")

        pose_diff = self.q_des - self.q
        prp_state = pose_diff
        prp_state = prp_state.reshape(1,-1)
        self.current_time += 1

        return prp_state, reward, done, info

    def get_reward(self):
        
        done = False
        pose_error = self.pose_error

        if not self.const_broken:

            if pose_error < 0.1:
                done = True
                reward = 10
            # elif pose_error < 1:
            #     done = True
            #     reward = 0
            else:
                reward = -(pose_error*10)

            if self.current_time > self.max_time:
                done = True
                reward -= 2
        
        else:
            reward = -100      
            done = True

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
        
        if np.any(self.q[:3] > self.max_q_bound[:3]) or np.any(self.q[:3] < self.min_q_bound[:3]):
            return True
        
        return False
    
    def get_error(self):

        pose_error =  np.linalg.norm(self.q - self.q_des) 

        return pose_error
        
    def reset(self,pose = np.array([0,0,2]),pose_des = None,max_time = 10):

        self.q = pose
        
        if pose_des is None:
            self.q_des = np.random.randint([-1,-1,1],[2,2,4])
        else:
            self.q_des = pose_des

        self.publish_simulator(self.q)
        pose_diff = self.q_des - self.q
        # pose_diff = np.clip(pose_diff,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = pose_diff
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.max_time = max_time

        return prp_state
    
    def publish_simulator(self,q):
        
        uav_pos_ort = list(q)[0:3]
        uav_pos_ort += [0,0,0]
        uav_pos_ort = f"{uav_pos_ort}"[1:-1]

        uav_msg = String()
        uav_msg.data = uav_pos_ort

        self.uam_publisher.send_request(uav_msg)

    def checkSelfContact(self):

        is_contact = False
        return is_contact     