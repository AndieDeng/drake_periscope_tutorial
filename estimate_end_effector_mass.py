import os.path
import numpy as np
import yaml

import pydrake

from pydrake.all import (
    RigidBodyFrame,
    RigidBodyTree,
    AddModelInstanceFromUrdfFile,
    FloatingBaseType,
)


class EstimateIiwaEeInertia:
    def __init__(self):
        self.iiwa_urdf_path = os.path.join(
            pydrake.getDrakePath(),
            "manipulation", "models", "iiwa_description", "urdf",
            "iiwa14_no_collision.urdf")

        self.iiwa_urdf_path_w_ee_mass = os.path.join(
                pydrake.getDrakePath(),
                "manipulation", "models", "iiwa_description", "urdf",
                "iiwa14_no_collision_w_ee_mass.urdf")
        self.tree = RigidBodyTree()

        self.robot_base_frame = RigidBodyFrame(
            "robot_base_frame", self.tree.world(), [0, 0, 0], [0, 0, 0])

        AddModelInstanceFromUrdfFile(self.iiwa_urdf_path, \
                                     FloatingBaseType.kFixed, self.robot_base_frame, self.tree)


        # print info about rigid body tree
        n_positions = self.tree.get_num_positions()
        for i in range(n_positions):
            print i, self.tree.get_position_name(i)

        n_bodies = self.tree.get_num_bodies()
        for i in range(n_bodies):
            print i, self.tree.getBodyOrFrameName(i)


    def CalcA(self, g_ee):
        A = np.zeros((6,4))
        A[0:3, 0:3] = -1*np.array(\
                        [[0, -g_ee[2], g_ee[1]], \
                         [g_ee[2], 0, -g_ee[0]], \
                         [-g_ee[1], g_ee[0], 0]])
        A[3:6, 3] = g_ee

        return A

    def CalcEeInertia(self, dict_joint_states):
        g = np.array([0, 0, -9.81])
        idx_base = 0
        idx_ee = 11

        n_positions = self.tree.get_num_positions()
        n_samples = len(dict_joint_states)
        An = np.zeros((n_samples * n_positions, 4))
        bn = np.zeros(n_samples * n_positions)

        i = 0
        for pose in dict_joint_states:
            q = np.array(dict_joint_states[pose]['joint_angles'])
            tau_external = np.array(dict_joint_states[pose]['torque_external'])

            kinsol = self.tree.doKinematics(q)
            J_ee, v_indices = self.tree.geometricJacobian(kinsol, idx_base, idx_ee, idx_ee)

            T_WE = self.tree.CalcBodyPoseInWorldFrame(kinsol, self.tree.FindBody('iiwa_link_ee'))
            R_WE = T_WE[0:3, 0:3]

            g_ee = R_WE.T.dot(g)
            An[n_positions*i:n_positions*(i+1), :] = J_ee.T.dot(self.CalcA(g_ee))
            bn[n_positions*i:n_positions*(i+1)] = tau_external

            i += 1

        print np.linalg.lstsq(An, bn, rcond=None)

    def GenerateFakeData(self, dict_joint_states):
        dict_fake = dict_joint_states.copy()
        tree_w_mass = RigidBodyTree()
        AddModelInstanceFromUrdfFile(self.iiwa_urdf_path_w_ee_mass, \
            FloatingBaseType.kFixed, self.robot_base_frame, tree_w_mass)

        for pose in dict_fake:
            q = np.array(dict_fake[pose]['joint_angles'])
            v = np.zeros(self.tree.get_num_velocities())
            kinsol_no_mass = self.tree.doKinematics(q, v)
            kinsol_w_mass = tree_w_mass.doKinematics(q, v)
            tau_w_mass = tree_w_mass.dynamicsBiasTerm(kinsol_w_mass, {})
            tau_no_mass = self.tree.dynamicsBiasTerm(kinsol_no_mass, {})


            dict_fake[pose]['torque_external'] = -(tau_w_mass - tau_no_mass)

        return dict_fake

if __name__ == '__main__':
    # opens data file
    data_file = open('ee_inertia_calibraton_data.yaml', 'r')
    dict_joint_states = yaml.load(data_file.read())

    estimate_inertia = EstimateIiwaEeInertia()
    estimate_inertia.CalcEeInertia(dict_joint_states)

    dict_fake = estimate_inertia.GenerateFakeData(dict_joint_states)
    estimate_inertia.CalcEeInertia(dict_fake)
