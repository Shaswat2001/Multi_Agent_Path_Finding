
import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import String 

from uam_msgs.srv import RequestUavPose
from uam_msgs.srv import ResponseUavPose
from uam_msgs.srv import RequestUavVel
from std_srvs.srv import Empty


class UavClientAsync(Node):

    def __init__(self):
        super().__init__('uam_client_async')
        self.cli = self.create_client(RequestUavPose, 'get_uav_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RequestUavPose.Request()

    def send_request(self, uav_pose):
        self.req.uav_pose = uav_pose
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

class ResetSimClientAsync(Node):

    def __init__(self):
        super().__init__('reset_sim_client_async')
        self.cli = self.create_client(RequestUavPose, 'get_uav_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RequestUavPose.Request()

    def send_request(self):
        uav_msg = String()
        uav_msg.data = "0, 0, 2, 0, 0, 0"
        self.req.uav_pose = uav_msg
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

class GetUavPoseClientAsync(Node):

    def __init__(self):
        super().__init__('get_uav_pose_client_async')
        self.cli = self.create_client(ResponseUavPose, 'send_uav_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = ResponseUavPose.Request()

    def send_request(self):
        self.req.get_pose = 1
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

        pose_uav = self.future.result().pose_uav.position
        return np.array([pose_uav.x,pose_uav.y,pose_uav.z])

class PauseGazeboClient(Node):

    def __init__(self):
        super().__init__('pause_gazebo_client')
        
        self.cli = self.create_client(Empty, '/pause_physics')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

    def send_request(self):
        self.cli.call(Empty.Request())

class UnPauseGazeboClient(Node):

    def __init__(self):
        super().__init__('unpause_gazebo_client')
        
        self.cli = self.create_client(Empty, '/unpause_physics')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

    def send_request(self):
        self.cli.call(Empty.Request())