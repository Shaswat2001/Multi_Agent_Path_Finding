import rclpy
import numpy as np
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')

        self.lidar_range = None
        self.ee_pose_subscription = self.create_subscription(LaserScan,"/LaserPlugin/out",self.lidar_callback,10)
        self.ee_pose_subscription  # prevent unused variable warning

    def get_state(self):

        if self.lidar_range is None:
            return np.zeros(shape=(360)),False
        
        lidar_data = np.array(self.lidar_range)
        contact = False
        for i in range(lidar_data.shape[0]):
            if lidar_data[i] == np.inf:
                lidar_data[i] = 1
            if lidar_data[i] < 0.1:
                contact = True

        return lidar_data,contact
    
    def lidar_callback(self, msg):

        self.lidar_range = msg.ranges

class CollisionSubscriber(Node):

    def __init__(self):
        super().__init__('collision_subscriber')

        self.contact_bumper_msg = None
        self.collision = False
        self.collision_subscription = self.create_subscription(ContactsState,"/bumper_states",self.collision_callback,10)
        self.collision_subscription  # prevent unused variable warning

    def get_collision_info(self):
    
        if self.collision:
            self.collision = False
            return True 
    
        return False

    def collision_callback(self, msg):
        self.contact_bumper_msg = msg

        if self.contact_bumper_msg is not None and len(self.contact_bumper_msg.states) != 0:
            self.collision = True

class StaticFramePublisher(Node):
    """
    Broadcast transforms that never change.

    This example publishes transforms from `world` to a static turtle frame.
    The transforms are only published once at startup, and are constant for all
    time.
    """

    def __init__(self):
        super().__init__('static_turtle_tf2_broadcaster')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

    def make_transforms(self, child,pose):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = child

        t.transform.translation.x = float(pose[0])
        t.transform.translation.y = float(pose[1])
        t.transform.translation.z = float(pose[2])
        quat = [0.0,0.0,0.0,1.0]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_static_broadcaster.sendTransform(t)

class PathPublisherDDPG(Node):

    def __init__(self):
        super().__init__('path_piblisher_ddpg')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_uav/ddpg', 1)
        self.path = Marker()
        self.path.type = self.path.LINE_STRIP
        self.path.action = self.path.ADD
        self.path.header.frame_id = "world"
        self.path.color.r = 0.5
        self.path.color.g = 0.2
        self.path.color.b = 0.0
        self.path.color.a = 1.0
        self.path.id = 10
        self.path.pose.orientation.w = 1.0
        self.path.scale.x = 0.065
        self.path.scale.y = 0.065
        self.path.scale.z = 0.065

    def add_point(self,pose_uav):
        pose = Point()
        pose.x = pose_uav[0]
        pose.y = pose_uav[1]
        pose.z = pose_uav[2]
        self.path.points.append(pose)

    def publish_robot(self):
        self.publisher_.publish(self.path)

class PathPublisherTD3(Node):

    def __init__(self):
        super().__init__('path_piblisher_td3')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_uav/td3', 10)
        self.path = Marker()
        self.path.type = self.path.LINE_STRIP
        self.path.action = self.path.ADD
        self.path.header.frame_id = "world"
        self.path.color.g = 1.0
        self.path.color.a = 1.0
        self.path.id = 10
        self.path.pose.orientation.w = 1.0
        self.path.scale.x = 0.065
        self.path.scale.y = 0.065
        self.path.scale.z = 0.065


    def add_point(self,pose_uav):
        pose = Point()
        pose.x = pose_uav[0]
        pose.y = pose_uav[1]
        pose.z = pose_uav[2]
        self.path.points.append(pose)

    def publish_robot(self):
        self.publisher_.publish(self.path)

class PathPublisherSAC(Node):

    def __init__(self):
        super().__init__('path_piblisher_sac')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_uav/sac', 10)
        self.path = Marker()
        self.path.type = self.path.LINE_STRIP
        self.path.action = self.path.ADD
        self.path.header.frame_id = "world"
        self.path.color.r = 1.0
        self.path.color.a = 1.0
        self.path.id = 10
        self.path.pose.orientation.w = 1.0
        self.path.scale.x = 0.065
        self.path.scale.y = 0.065
        self.path.scale.z = 0.065


    def add_point(self,pose_uav):
        pose = Point()
        pose.x = pose_uav[0]
        pose.y = pose_uav[1]
        pose.z = pose_uav[2]
        self.path.points.append(pose)

    def publish_robot(self):
        self.publisher_.publish(self.path)

class PathPublisherSoftQ(Node):

    def __init__(self):
        super().__init__('path_piblisher_softq')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_uav/softq', 10)
        self.path = Marker()
        self.path.type = self.path.LINE_STRIP
        self.path.action = self.path.ADD
        self.path.header.frame_id = "world"
        self.path.color.b = 1.0
        self.path.color.a = 1.0
        self.path.id = 10
        self.path.pose.orientation.w = 1.0
        self.path.scale.x = 0.065
        self.path.scale.y = 0.065
        self.path.scale.z = 0.065


    def add_point(self,pose_uav):
        pose = Point()
        pose.x = pose_uav[0]
        pose.y = pose_uav[1]
        pose.z = pose_uav[2]
        self.path.points.append(pose)

    def publish_robot(self):
        self.publisher_.publish(self.path)