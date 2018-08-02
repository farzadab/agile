'''
Contains the specifications for the robots used in environments
'''
import numpy as np
import copy
import gym
import os

from .robot_locomotors import WalkerBase
from algorithms.plot import LinePlot

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class Walker2D(WalkerBase):
    '''
    Just the 2D Walker model from PyBullet:
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/robot_locomotors.py
    '''
    ee_names = foot_list = ["foot", "foot_left"]
    model_filename = "models/walker2d.xml"
    pelvis_partname = 'link0_6'

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
            self.get_pelvis_position(),
            self.get_joint_positions(),
        ])

    def get_pelvis_position(self):
        return self.pelvis.current_position()

    def reset(self, bullet_client):
        super().reset(bullet_client)
        self.ordered_part_names = sorted([k for k in self.parts if k[:4] != 'link' and k != 'floor'])
        self.ordered_parts = [self.parts[name] for name in self.ordered_part_names]
        self.pelvis = self.parts[self.pelvis_partname]
        self.correction = np.array(self.robot_body.current_position() - self.pelvis.current_position())
        self.correction[2] -= self.initial_z

    def get_pelvis_velocity(self):
        return self.pelvis.speed()

    def get_joint_positions(self):
        return [joint.get_position() for joint in self.ordered_joints]

    def get_joint_velocities(self):
        return [joint.get_velocity() for joint in self.ordered_joints]

    def get_end_eff_positions(self):
        root = self.pelvis.current_position()
        return np.concatenate([
            np.subtract(self.parts[ee_name].current_position(), root)
            for ee_name in self.end_effector_names
        ])

    def get_part_positions(self):
        return np.concatenate([
            self.parts[p_name].current_position()
            for p_name in self.ordered_part_names
        ])

    def reset_stationary_pose(self, root_position, joint_positions, root_velocity=[0, 0, 0], joint_velocities=None, limb_velocities=None):
        assert len(self.ordered_joints) == len(joint_positions)
        # root_position = copy.copy(root_position)
        if joint_velocities is None:
            joint_velocities = [0] * len(joint_positions)

        if limb_velocities is None or len(limb_velocities) == 0:
            limb_velocities = [0] * (3 * len(self.ordered_part_names))
        
        for i, part in enumerate(self.ordered_parts):
            part.reset_velocity(linearVelocity=limb_velocities[3*i:3*(i+1)])

        # root_position[2] += 0.5#j.current_relative_position()
        # root_position[2] -= self.initial_z

        # self.pelvis.reset_pose(root_position + self.correction, [0.000000,	0.09983341664682815, 0.0,  0.9950041652780258,])
        # self.pelvis.reset_pose(root_position + self.correction, [0.000000,	0.137993, 0 , 0.990433])
        self.pelvis.reset_pose(root_position + self.correction, [0, 0, 0, 1])
        self.pelvis.reset_velocity(linearVelocity=root_velocity)
        # self.robot_body.reset_velocity(linearVelocity=root_velocity)
        self.body_xyz = root_position

        for i, joint in enumerate(self.ordered_joints):
            joint.reset_position(joint_positions[i], joint_velocities[i])


class WalkerV2(Walker2D):
    '''Walker2D with fixed thigh joint ranges'''
    model_filename = 'models/walker_v2.xml'

    # def alive_bonus(self, z, pitch):
    #     return 1


class Walker2DNoMass(WalkerV2):
    def __init__(self):
        super().__init__()

    def reset(self, bullet_client):
        super().reset(bullet_client)
        # for name, part in self.parts.items():
        #     print(name, bullet_client.getDynamicsInfo(part.bodies[part.bodyIndex], part.bodyPartIndex)[0])
        for part in self.parts.values():
            bullet_client.changeDynamics(part.bodies[part.bodyIndex], part.bodyPartIndex, mass=0)


class Walker2DPD(WalkerV2):
    kp = 2
    kd = 2
    def __init__(self):
        super().__init__()
        if self._debug:
            self.plot = LinePlot(xlim=[0, 1000], ylim=[-10, 10])
            self.plot2 = LinePlot(xlim=[0, 1000], ylim=[-10, 10])
            self.plot3 = LinePlot(xlim=[0, 1000], ylim=[-10, 10])
            self.plot4 = LinePlot(xlim=[0, 1000], ylim=[-10, 10])

    def apply_action(self, a):
        joint_pose = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32
        )
        action = self.kp * (a - joint_pose[:, 0]) - self.kd * joint_pose[:, 1]
        if self._debug:
            self.plot.add_point(joint_pose[0, 1], self._istep)
            self.plot2.add_point(joint_pose[0, 0], self._istep)
            self.plot3.add_point(a[0], self._istep)
        super().apply_action(action)


class TRLWalker(Walker2DPD):
    part_names = ['thigh', 'leg', 'foot', 'thigh_left', 'leg_left', 'foot_left']
    def __init__(self):
        super().__init__()
        obs_dim = len(self.part_names) * 6 + 3 + 4
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

    def calc_state(self):
        super().calc_state()  # this function has some side-effects, so let's keep it for now

        pelvis_state = self.parts[self.pelvis_partname].get_pose()
        parts_state = np.array([
            (
                np.subtract(self.parts[p].current_position(), pelvis_state[:3]),
                self.parts[p].speed(),
            )
            for p in self.part_names
        ]).flatten()

        return np.concatenate([pelvis_state, parts_state])


class FixedWalker(WalkerV2):
    model_filename = "models/fixed_walker.xml"

    def reset_stationary_pose(self, root_position, joint_positions, root_velocity=[0, 0, 0], joint_velocities=None):
        assert len(self.ordered_joints) == len(joint_positions)

        joint_velocities = [0] * len(joint_positions)

        root_position[2] += 0.4
        root_position[0] = 0
        self.robot_body.reset_velocity(linearVelocity=[0, 0, 0])

        # self.robot_body.reset_pose(root_position, [0.000000,	0.137993, 0, 0.990433])
        self.robot_body.reset_pose(root_position, [0, 0, 0, 1])
        self.body_xyz = root_position

        for part in self.parts.values():
            part.reset_velocity()

        for i, joint in enumerate(self.ordered_joints):
            joint.reset_position(joint_positions[i], joint_velocities[i])


class FixedPDWalker(Walker2DPD):
    model_filename = "models/fixed_walker.xml"
    pelvis_partname = 'link0_4'

    def reset_stationary_pose(self, root_position, joint_positions, root_velocity=[0, 0, 0], joint_velocities=None):
        FixedWalker.reset_stationary_pose(
            self,
            root_position, joint_positions, root_velocity, joint_velocities
        )
