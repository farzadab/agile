'''
Contains the specifications for the robots used in environments
'''
import numpy as np
import os

from .robot_locomotors import WalkerBase


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class Walker2D(WalkerBase):
    '''
    Just the 2D Walker model from PyBullet:
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/robot_locomotors.py
    '''
    foot_list = ["foot", "foot_left"]
    model_filename = "models/walker2d.xml"

    def __init__(self):
        self.end_effector_names = self.foot_list
        super().__init__(
            os.path.join(CUR_DIR, self.model_filename),
            "torso",
            action_dim=6, obs_dim=22, power=0.40
        )

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

    def robot_specific_reset(self, bullet_client):
        super().robot_specific_reset(bullet_client)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0

    def get_stationary_pose(self):
        return np.concatenate([
            self.get_com_position(),
            self.get_joint_positions(),
        ])

    def get_com_position(self):
        com = self.robot_body.current_position()
        com[2] -= self.initial_z
        return com

    def get_joint_positions(self):
        return [joint.get_position() for joint in self.ordered_joints]

    def get_joint_velocities(self):
        return [joint.get_velocity() for joint in self.ordered_joints]

    def get_end_eff_positions(self):
        return np.concatenate([
            self.parts[ee_name].current_position() for ee_name in self.end_effector_names
        ])

    def reset_stationary_pose(self, root_position, joint_positions, root_velocity=[0, 0, 0], joint_velocities=None):
        assert len(self.ordered_joints) == len(joint_positions)

        if joint_velocities is None:
            joint_velocities = [0] * len(joint_positions)

        self.robot_body.reset_pose(root_position, [0, 0, 0, 1])
        self.robot_body.reset_velocity(linearVelocity=root_velocity)
        self.body_xyz = root_position

        for part in self.parts.values():
            part.reset_velocity()

        for i, joint in enumerate(self.ordered_joints):
            joint.reset_position(joint_positions[i], joint_velocities[i])


class Walker2DNoMass(Walker2D):
    def __init__(self):
        super().__init__()

    def reset(self, bullet_client):
        super().reset(bullet_client)
        for part in self.parts.values():
            bullet_client.changeDynamics(part.bodies[part.bodyIndex], part.bodyPartIndex, mass=0)
