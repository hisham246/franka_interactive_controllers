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

    rotation_matrix_d = r.as_matrix()


    if model == None:
        model = rbdl.loadModel("/home/hisham246/uwaterloo/panda_ws/src/franka_interactive_controllers/urdf/panda/panda.urdf")
    else:
        model = model

    constraints = rbdl.InverseKinematicsConstraintSet()

    rotation_matrix_d_col_major = rotation_matrix_d.transpose().reshape((3,3), order='C').copy()

    constraints.AddOrientationConstraint(np.array([model.GetBodyId('panda_hand_tcp')]), rotation_matrix_d_col_major)
    constraints.AddPointConstraint(np.array([model.GetBodyId('panda_hand_tcp')]), np.array([0., 0., -0.05]), position_d, weight=1)

    response = np.ndarray(shape=(7,))

    success = rbdl.InverseKinematicsCS(model, q_init, constraints, response)

    joint_mins = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_maxs = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    response = response[:7]


    return response
