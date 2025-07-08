#!/usr/bin/env python3

import rospy
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal

def move_gripper(width=0.04, speed=0.05):
    rospy.init_node('franka_gripper_test_node')

    client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
    rospy.loginfo("Waiting for /franka_gripper/move action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to gripper action server.")

    goal = MoveGoal(width=width, speed=speed)
    rospy.loginfo(f"Sending move goal: width={width} m, speed={speed} m/s")
    client.send_goal(goal)
    client.wait_for_result()

    result = client.get_result()
    if result and result.success:
        rospy.loginfo("Gripper move successful.")
    else:
        rospy.logwarn("Gripper move failed.")

if __name__ == "__main__":
    try:
        move_gripper(width=0.06, speed=0.03)  # Change width and speed as needed
    except rospy.ROSInterruptException:
        pass
