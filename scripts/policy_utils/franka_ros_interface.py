import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from dynamic_reconfigure.client import Client as DynClient
from scipy.spatial.transform import Rotation as R

class FrankaROSInterface:
    def __init__(self,
                #  ros_init=True,
                 pose_topic='/cartesian_pose_impedance_controller/desired_pose',
                 impedance_config_ns='/cartesian_pose_impedance_controller/dynamic_reconfigure_compliance_param_node',
                 ee_state_topic='/franka_state_controller/ee_pose',
                 joint_state_topic='/joint_states'):
        
        # if ros_init and not rospy.core.is_initialized():
        #     rospy.init_node("franka_ros_interface_node", anonymous=True)

        self.pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
        self.dyn_client = DynClient(impedance_config_ns)

        self.ee_pose = None
        self.joint_positions = None
        rospy.Subscriber(ee_state_topic, Pose, self._ee_pose_callback)
        rospy.Subscriber(joint_state_topic, JointState, self._joint_state_callback)

    def _ee_pose_callback(self, msg):
        pos = msg.position
        quat = msg.orientation
        rotvec = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_rotvec()
        self.ee_pose = np.array([pos.x, pos.y, pos.z, *rotvec])

    def _joint_state_callback(self, msg):
        self.joint_positions = np.array(msg.position)

    def get_ee_pose(self):
        while self.ee_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return self.ee_pose

    def get_joint_positions(self):
        while self.joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return self.joint_positions

    def update_desired_ee_pose(self, pose: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'panda_link0'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        quat = R.from_rotvec(pose[3:]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        self.pose_pub.publish(msg)

    def update_impedance_gains(self, Kx: np.ndarray, Kxd: np.ndarray):
        config = {
            'translational_stiffness_x': Kx[0],
            'translational_stiffness_y': Kx[1],
            'translational_stiffness_z': Kx[2],
            'rotational_stiffness_x': Kx[3],
            'rotational_stiffness_y': Kx[4],
            'rotational_stiffness_z': Kx[5],
            'translational_damping_x': Kxd[0],
            'translational_damping_y': Kxd[1],
            'translational_damping_z': Kxd[2],
            'rotational_damping_x': Kxd[3],
            'rotational_damping_y': Kxd[4],
            'rotational_damping_z': Kxd[5],
        }
        self.dyn_client.update_configuration(config)

    def terminate_policy(self):
        rospy.loginfo("[FrankaROSInterface] Terminating policy...")
