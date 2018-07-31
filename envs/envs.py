import numpy as np
from collections import OrderedDict
import gym
import time
import copy

from .robots import WalkerV2, Walker2DNoMass, Walker2DPD, FixedWalker, FixedPDWalker, TRLWalker
from .modified_base_envs import WalkerBaseBulletEnv
from .walker_paths import WalkingPath, FastWalkingPath, TRLWalk, TRLStep, TRLRun
from .paths import RefMotionStore


class Walker2DEnv(WalkerBaseBulletEnv):
    def __init__(self, robot=None):
        self.robot = WalkerV2() if robot is None else robot
        WalkerBaseBulletEnv.__init__(self, self.robot)


class Walker2DRefEnv(Walker2DEnv):
    default_store_fname = 'walker.json'
    def __init__(self, rsi=True, ref=WalkingPath, robot=None, ref_robot=None, et_rew=0.0, et_com=0.5):
        '''
        @param rsi: whether or not to do Random-Start-Initialization (default: True)
        @param ref: the reference (kinematic) motion
        @param robot: the model to use in simulation
        @param et_rew: the reward threshold used for early termination. set `et_rew=0` to get rid of it
        @param et_rew: a for early termination based on the difference in CoM position. set `et_com=np.inf` to get rid of it
        '''
        self.timer = 0
        self.rsi = rsi
        self.et_rew = et_rew
        self.et_com = et_com
        self.ref_robot = Walker2DNoMass() if ref_robot is None else ref_robot
        super().__init__(robot=robot)
        self.ref = ref
        self.istep = 0
        self.correct_state_id = False
        high = self.observation_space.high
        low = self.observation_space.low
        self.observation_space = gym.spaces.Box(
            np.concatenate([low, [0]]),
            np.concatenate([high, [self.ref.one_cycle_duration()]]),
            dtype=np.float32,
        )

    def play_path(self, timesteps=None, store_fname=None):
        'Replays the reference trajectory + can store the data for later use'
        if store_fname is '':
            store_fname = self.default_store_fname

        self._reset()

        if timesteps is None:
            timesteps = int(np.round(self.ref.one_cycle_duration() * 10 / self.scene.dt))

        store = None
        if store_fname:
            store = RefMotionStore(
                dt=self.scene.dt,
                joint_names=[joint.joint_name for joint in self.robot.ordered_joints],
                end_eff_names=self.robot.ee_names
            )

        for _ in range(timesteps):
            pose = self.ref.pose_at_time(self.timer)
            self.reset_stationary_pose(pose)
            # self.robot.body_xyz = pose[:3]  # just a hack to make the camera_adjust work
            self.camera_adjust(4, 0, 10)

            self.timer += self.scene.dt
            time.sleep(self.scene.dt)
            if store:
                ee_local_pos = self.robot.get_end_eff_positions()
                store.record(
                    pose[3:],
                    ee_local_pos,
                    pose[:3],
                )

        if store:
            store.store(store_fname)
        
    def _seed(self, seed=None):
        ret = super()._seed(seed)
        self.ref_robot.np_random = self.np_random
        return ret
    
    def reset_ref_pose(self, pose):
        pose = copy.copy(pose)
        if self.isRender:
            pose[1] = 1
            self.ref_robot.reset_stationary_pose(pose[:3], pose[3:])

    def reset_stationary_pose(self, pose, vel=None):
        if vel is None:
            vel = [0] * 9
        self.robot.reset_stationary_pose(pose[:3], pose[3:], root_velocity=vel[:3], joint_velocities=vel[3:])
        self.reset_ref_pose(pose)

    def _reset(self):
        super()._reset()
        self.istep = 0

        self.dt = self.scene.dt
        # self._p.setGravity(0.0,0.0,0.0)
        if self.isRender:
            self.ref_robot.scene = self.scene
            self.ref_robot.reset(self._p)

        if not self.correct_state_id:
            self.stateId = self._p.saveState()
            self.correct_state_id = True

        if self.rsi:
            self.timer = self.np_random.uniform(0, self.ref.one_cycle_duration())
        else:
            self.timer = 0

        pose = self.ref.pose_at_time(self.timer)
        vel = self.ref.vel_at_time(self.timer)
        self.reset_stationary_pose(pose, vel)

        self.camera_adjust(4)

        return self.get_obs_with_phase()
    
    def get_reward(self, state, action):
        # weights = np.array([15, 0, 2, 1, 1, 1, 1, 1, 1])  # mean actually gives us: [10/9, 0/9, 2/9 ....]
        # diff = np.exp(-.05 * (ref_pose - pose) ** 2)
        # ref_reward = 64 * np.mean(weights * diff)

        ref_pose = self.ref.pose_at_time(self.timer)
        pose = self.robot.get_stationary_pose()

        weights = np.array([0.2, 0, 2, 1, 1, 1, 1, 1, 1])  # mean actually gives us: [10/9, 0/9, 2/9 ....]
        diff = (ref_pose - pose) ** 2
        ref_reward = -1 * np.mean(weights * diff)

        self.rewards['ref_pose'] = ref_reward

        rew = sum(self.rewards.values())

        rew += -1 * self.rewards['progress']
        self.rewards['progress'] *= 4

        return rew

    def get_obs_with_phase(self, robot_state=None):
        if robot_state is None:
            robot_state = self.robot.calc_state()
        phase = self.timer % self.ref.one_cycle_duration()
        return np.concatenate([robot_state, [phase]])

    def action_transform(self, action):
        return action

    def _step(self, action):
        self.timer += self.scene.dt

        if self.isRender:
            time.sleep(self.scene.dt)

        obs, _, done, extra = super()._step(self.action_transform(action))
        self.istep += 1
        self.rewards = extra.get('rewards', {})

        ref_pose = self.ref.pose_at_time(self.timer)
        self.reset_ref_pose(ref_pose)

        rew = self.get_reward(obs, action)
        extra['rewards'] = self.rewards

        if np.sum(np.subtract(ref_pose[:3], self.robot.get_pelvis_position()) ** 2) > self.et_com:
            done = True
            extra['termination'] = 'com'
        elif rew < self.et_rew:
            done = True
            extra['termination'] = 'rew'

        return self.get_obs_with_phase(obs), rew, done, extra


class Walker2DRefEnvDM(Walker2DRefEnv):
    '''
    Walker2DRef environment with the corrected rewards
    '''
    r_names = ['jpos', 'jvel', 'ee', 'pelvis_z', 'pelvis_v']
    r_weights = dict(jpos=0.4 , jvel=0.1, ee=0.1 , pelvis_z=0.02, pelvis_v=0.38)
    r_scales  = dict(jpos=2   , jvel=0.1, ee=40/3, pelvis_z=10/3, pelvis_v=10)
    def __init__(self, store_fname=None, ref=None, **kwargs):
        if store_fname is None:
            store_fname = self.default_store_fname
        if ref is None:
            ref = RefMotionStore().load(store_fname)
        super().__init__(ref=ref, **kwargs)

    def cur_motion_params(self):
        pelvis_p = self.robot.get_pelvis_position()
        return {
            'jpos': self.robot.get_joint_positions(),
            'jvel': self.robot.get_joint_velocities(),
            'ee': self.robot.get_end_eff_positions(),
            'pelvis': pelvis_p,
            'pelvis_z': pelvis_p[2],
            'pelvis_v': self.robot.get_pelvis_velocity(),
        }

    def get_reward(self, state, action):
        targets = self.ref.ref_at_time(self.timer)
        current = self.cur_motion_params()
        self.rewards = OrderedDict([
            (param,
                np.exp(
                    -1 * self.r_scales[param] * np.sum(np.square(
                        np.subtract(targets[param], current[param])
                    ))
                )
            ) for param in self.r_names
        ])
        # print(current['jpos'], targets['jpos'], self.rewards['jpos'])
        # print(current['ee'], targets['ee'], self.rewards['ee'])
        # print(current['jvel'], targets['jvel'], self.rewards['jvel'])
        # print(current['pelvis'], targets['pelvis'], self.rewards['pelvis'])
        # print(current['pelvis_v'], targets['pelvis_v'], self.rewards['pelvis_v'])
        # print(current['pelvis_z'], targets['pelvis_z'], self.rewards['pelvis_z'])
        # print(self.rewards.values())
        return sum([self.r_weights[param] * self.rewards[param] for param in self.r_names])


class FastWalker2DRefEnvDM(Walker2DRefEnvDM):
    default_store_fname = 'fast_walker.json'


class Walker2DPDRefEnvDM(Walker2DRefEnvDM):
    def __init__(self, robot=Walker2DPD()):
        super().__init__(robot=robot)


class FixedWalkerRefEnvDM(Walker2DRefEnvDM):
    r_names = ['jpos', 'jvel', 'ee']
    def __init__(self, robot=None):
        super().__init__(robot=robot or FixedWalker(), ref_robot=FixedWalker())


class FixedWalker2DPDRefEnvDM(FixedWalkerRefEnvDM):
    def __init__(self):
        super().__init__(robot=FixedPDWalker())


class FixedSlowRunnerPDRefEnvDM(FixedWalkerRefEnvDM):
    default_store_fname = '2d_run_slower.json'
    def __init__(self):
        super().__init__(robot=FixedPDWalker())

class FixedSlowerRunnerPDRefEnvDM(FixedWalkerRefEnvDM):
    default_store_fname = '2d_run_slower1.5.json'
    def __init__(self):
        super().__init__(robot=FixedPDWalker())


class FixedRunnerPDRefEnvDM(FixedWalkerRefEnvDM):
    default_store_fname = '2d_run.json'
    def __init__(self):
        super().__init__(robot=FixedPDWalker())


class TRLRunBadEnvDM(Walker2DRefEnvDM):
    '''
    The motion is actually flawed: the torso should be leaning forward but it isn't in this motion
    '''
    default_store_fname = '2d_run_bad.json'


class TRLRunBadPDEnvDM(Walker2DPDRefEnvDM):
    '''
    The motion is actually flawed: the torso should be leaning forward but it isn't in this motion
    '''
    default_store_fname = '2d_run_bad.json'


class TRLRunPDEnvDM(Walker2DPDRefEnvDM):
    '''
    2D biped running motion
    '''
    default_store_fname = '2d_run.json'


class TRLWalkerPDEnvDM(Walker2DRefEnvDM):
    def __init__(self, robot=None):
        super().__init__(robot=robot or TRLWalker())


class TRLSlowRunPDEnvDM(TRLWalkerPDEnvDM):
    '''
    2D biped running motion (1.3 times slower than original)
    '''
    default_store_fname = '2d_run_slower.json'


def display_robot_parts_with_cube():
    from envs.robot_locomotors import get_cube

    env = FixedWalker2DPDRefEnvDM()
    env.render('human')
    env._reset()
    env._p.setGravity(0.0, 0.0, 0.0)
    part_names = list(env.robot.parts.keys())
    cube = get_cube(env._p, 0, 0, 0)
    print(part_names)
    env._step(env.ref.pose_at_time(env.timer)[3:])
    for i in range(1000):
        # env.timer += env.scene.dt
        # env.parts[env.robot.pelvis_partname].reset_pose(env.ref.pose_at_time(env.timer)[:3], [.1, 0, 0, 1])
        part = env.robot.parts[part_names[i % len(part_names)]]
        print(part_names[i % len(part_names)])
        pos = part.current_position()
        pos[1] -= 0.1
        cube.reset_position(pos)
        # print(env.robot.robot_body.current_position(), env.parts[env.robot.pelvis_partname].current_position())
        time.sleep(env.scene.dt * 100)


def pd_drive_fixed_walker():
    env = Walker2DPDRefEnvDM()
    env.render('human')
    env._reset()
    env._p.setGravity(0.0, 0.0, 0.0)
    for i in range(1000):
        env._step(env.ref.pose_at_time(env.timer)[3:])
        time.sleep(env.scene.dt)


def play_path(env, record=False, ref=None):
    if ref is not None:
        env.ref = ref
    env.render('human')
    env.play_path(store_fname='' if record else None)
    # env.play_path(store_fname='2d_run_slower1.5.json' if record else None)


if __name__ == '__main__':
    play_path(Walker2DRefEnv(), False, ref=TRLRun)
    # pd_drive_fixed_walker()
    # display_robot_parts_with_cube()
