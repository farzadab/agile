'''
A modified version of `WalkerBaseBulletEnv` from: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_locomotion_envs.py
'''
import pybullet
import numpy as np
from collections import OrderedDict

from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv


class WalkerBaseBulletEnv(MJCFBaseBulletEnv):

    def __init__(self, robot, render=False):
        print("WalkerBase::__init__ start")
        MJCFBaseBulletEnv.__init__(self, robot, render)

        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1
        self.debug = False

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(
            bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4
        )
        return self.stadium_scene

    def _reset(self):
        if self.stateId >= 0:
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv._reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
            self._p, self.stadium_scene.ground_plane_mjcf
        )
        self.ground_ids = set(
            [
                (
                    self.parts[f].bodies[self.parts[f].bodyIndex],
                    self.parts[f].bodyPartIndex,
                )
                for f in self.foot_ground_object_names
            ]
        )
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        # print("saving state self.stateId:",self.stateId)

        return r

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(
            init_x, init_y, init_z
        )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost = (
        -2.0
    )  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = (
        -0.1
    )  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = (
        -1.0
    )  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def _step(self, a):
        if (
            not self.scene.multiplayer
        ):  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z, self.robot.body_rpy[1]
            )
        )  # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
            self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                if self.isRender and self.debug:
                    [
                        self._p.addUserDebugLine(
                            x,
                            x + y*z/100,
                            lineColorRGB=(1, 0, 0),
                            lineWidth=5,
                            lifeTime=1. / 60.,
                        )
                        for x, y, z in map(
                            lambda x: (np.array(x[6]), np.array(x[7]), x[9]),
                            f.contact_list(),
                        )
                    ]
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(
            np.abs(a * self.robot.joint_speeds).mean()
        )  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
        ]

        rewards_dict = OrderedDict([
            ('alive', alive),
            ('progress', progress),
            ('electricity', electricity_cost),
            ('joints_at_limit', joints_at_limit_cost),
        ])

        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))

        keys = self._p.getKeyboardEvents()
        if ord('1') in keys and keys[ord('1')] == self._p.KEY_WAS_RELEASED:
            # keys is a dict, so need to check key exists
            self.debug = not self.debug

        if self.isRender and self.debug:
            x, y, z = self.robot.body_xyz
            vx, vy, vz = self.robot_body.speed()
            self._p.addUserDebugLine(
                (x,y,z),
                (x+vx, y+vy, z+vz),
                lineColorRGB=(0, 0, 1),
                lineWidth=5,
                lifeTime=1. / 60.,
            )


        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {'rewards': rewards_dict}

    def camera_adjust(self, distance=10, yaw=10):
        x, y, z = self.robot.body_xyz
        self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        # self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)
        lookat = [x, y, z]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)
    
    def get_stationary_pose(self):
        return np.concatenate([
            self.robot_body.current_position(),
            [joint.get_position() for joint in self.robot.ordered_joints],
        ])
    
    def reset_stationary_pose(self, root_position, joint_positions):
        assert(len(self.robot.ordered_joints) == len(joint_positions))

        self.robot_body.reset_pose(root_position, [0, 0, 0, 1])
        self.robot.body_xyz = root_position
        for i, joint in enumerate(self.robot.ordered_joints):
            joint.reset_position(joint_positions[i], 0)
