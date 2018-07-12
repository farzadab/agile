'''
Contains the specifications for the robots used in environments
'''
import os

from .robot_locomotors import WalkerBase


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class Walker2D(WalkerBase):
    '''
    Just the 2D Walker model from PyBullet:
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/robot_locomotors.py
    '''
    foot_list = ["foot", "foot_left"]

    def __init__(self):
        super().__init__(
            os.path.join(CUR_DIR, "models/walker2d.xml"),
            "torso",
            action_dim=6, obs_dim=22, power=0.40
        )

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

    def robot_specific_reset(self, bullet_client):
        super().robot_specific_reset(bullet_client)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0
