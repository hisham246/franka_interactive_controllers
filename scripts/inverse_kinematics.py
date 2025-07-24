import rbdl
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math



def InverseKin(q_init, position_d, orientation_d, model=None):
    """
    args:
        q_init: the inital joint states
        position_d: numpy array of desired positions
        orientation_d: numpy array with desired orientation.  quaternion with [w, x, y, z]
    """

    orientation_d = orientation_d[[1, 2, 3, 0]]


    r = R.from_quat(orientation_d)

    # rotation_matrix_d = r.as_matrix()
    rotation_matrix_d = r.as_dcm()

    # print('initial rotation matrix', rotation_matrix_d[0, 0], rotation_matrix_d[0, 1], rotation_matrix_d[0, 2])

    if model == None:
        model = rbdl.loadModel("/home/robohub/ros_ws/src/lfd/panda_control/urdf/panda/panda.urdf")
        # model = rbdl.loadModel("/home/airlab4/ros_ws/src/LfD/panda_control/urdf/panda/panda.urdf")
    else:
        model = model


    # print('model bodies', model.GetBodyId('panda_hand_tcp'))

    constraints = rbdl.InverseKinematicsConstraintSet()
    # constraints = rbdl.InverseKinematicsConstraintSet(constraint_tol=1e-5)

    # print("2222")
    rotation_matrix_d_col_major = rotation_matrix_d.transpose().reshape((3,3), order='C').copy()

    # constraints.AddFullConstraint(np.array([8]), np.zeros(shape=(3)), position_d, rotation_matrix_d)
    # constraints.AddFullConstraint(np.array([model.GetBodyId('panda_hand_tcp')]), np.zeros(shape=(3)), position_d, rotation_matrix_d)
    constraints.AddOrientationConstraint(np.array([model.GetBodyId('panda_hand_tcp')]), rotation_matrix_d_col_major)
    constraints.AddPointConstraint(np.array([model.GetBodyId('panda_hand_tcp')]), np.array([0., 0., -0.05]), position_d, weight=1)

    response = np.ndarray(shape=(7,))

    success = rbdl.InverseKinematicsCS(model, q_init, constraints, response)

    joint_mins = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_maxs = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    response = response[:7]

    # for i in range(7):
    #     response[i] = LimitJoint(joint_mins[i], joint_maxs[i], response[i])

    # print(success)
    # print(response)


    return response[:7]
