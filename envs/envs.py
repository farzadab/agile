import numpy as np
from collections import OrderedDict
import gym
import copy

from .robots import WalkerV2, Walker2DNoMass, Walker2DPD, FixedWalker, FixedPDWalker
from .modified_base_envs import WalkerBaseBulletEnv
from .walker_paths import WalkingPath, FastWalkingPath
from .paths import RefMotionStore


class Walker2DEnv(WalkerBaseBulletEnv):
    def __init__(self, robot=None):
        self.robot = WalkerV2() if robot is None else robot
        WalkerBaseBulletEnv.__init__(self, self.robot)


class Walker2DRefEnv(Walker2DEnv):
    default_store_fname = 'walker.json'
    def __init__(self, rsi=True, ref=WalkingPath, robot=None, et_rew=0):
        '''
        @param rsi: whether or not to do Random-Start-Initialization (default: True)
        @param ref: the reference (kinematic) motion
        @param robot: the model to use in simulation
        @param et_rew: the reward threshold used for early termination. set `et_rew=0` to get rid of it
        '''
        self.timer = 0
        self.rsi = rsi
        self.et_rew = et_rew
        self.ref_robot = Walker2DNoMass()
        super().__init__(robot=robot)
        self.ref = ref
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
        if store_fname is None:
            store_fname = self.default_store_fname

        import time
        self._reset()

        if timesteps is None:
            timesteps = int(np.round(self.ref.one_cycle_duration() * 10 / self.scene.dt))

        store = None
        if store_fname:
            store = RefMotionStore(
                dt=self.scene.dt,
                joint_names=[joint.joint_name for joint in env.ref_robot.ordered_joints],
                end_eff_names=['foot', 'foot_left']
            )
        
        def fix_y(pos):
            pos[1] -= 1
            return pos

        for _ in range(timesteps):
            pose = self.ref.pose_at_time(self.timer)
            self.reset_ref_pose(pose)
            self.robot.body_xyz = pose[:3]  # just a hack to make the camera_adjust work
            self.camera_adjust(2)

            self.timer += self.scene.dt
            time.sleep(self.scene.dt)
            if store:
                vel = self.ref.vel_at_time(self.timer)
                ee_pos = np.concatenate([
                    fix_y(self.ref_robot.parts[ee_name].current_position())
                    for ee_name in store.end_eff_names
                ])
                store.record(
                    pose[3:],
                    vel[3:],
                    ee_pos,
                    pose[:3],
                )

        # if store:
        #     store.store(store_fname)
        
    def _seed(self, seed=None):
        ret = super()._seed(seed)
        self.ref_robot.np_random = self.np_random
        return ret
    
    def reset_ref_pose(self, pose):
        pose = copy.copy(pose)
        if self.isRender:
            pose[1] = 1
            self.ref_robot.reset_stationary_pose(pose[:3], pose[3:])

    def reset_stationary_pose(self, pose, vel):
        self.robot.reset_stationary_pose(pose[:3], pose[3:], root_velocity=vel[:3], joint_velocities=vel[3:])
        self.reset_ref_pose(pose)

    def _reset(self):
        super()._reset()
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
        obs, _, done, extra = super()._step(self.action_transform(action))
        self.rewards = extra.get('rewards', {})

        ref_pose = self.ref.pose_at_time(self.timer)
        self.reset_ref_pose(ref_pose)

        rew = self.get_reward(obs, action)
        extra['rewards'] = self.rewards

        if rew < self.et_rew:
            done = True

        return self.get_obs_with_phase(obs), rew, done, extra


class Walker2DRefEnvDM(Walker2DRefEnv):
    '''
    Walker2DRef environment with the corrected rewards
    '''
    r_names = ['jpos', 'jvel', 'ee', 'torso_z', 'torso_v']
    r_weights = dict(jpos=0.65, jvel=0.1, ee=0.1 , torso_z=0.075, torso_v=0.075)
    r_scales  = dict(jpos=2   , jvel=0.1, ee=40/3, torso_z=10/3 , torso_v=10/3 )
    def __init__(self, store_fname=None, ref=None, **kwargs):
        if store_fname is None:
            store_fname = self.default_store_fname
        if ref is None:
            ref = RefMotionStore().load(store_fname)
        super().__init__(ref=ref, **kwargs)

    def cur_motion_params(self):
        torso_p = self.robot.get_torso_position()
        return {
            'jpos': self.robot.get_joint_positions(),
            'jvel': self.robot.get_joint_velocities(),
            'ee': self.robot.get_end_eff_positions(),
            'torso': torso_p,
            'torso_z': torso_p[2],
            'torso_v': self.robot.get_torso_velocity(),
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
        # print(current['torso'], targets['torso'], self.rewards['torso'])
        # print(current['torso_v'], targets['torso_v'], self.rewards['torso_v'])
        # print(self.rewards.values())
        return sum([self.r_weights[param] * self.rewards[param] for param in self.r_names])


class FastWalker2DRefEnvDM(Walker2DRefEnvDM):
    default_store_fname = 'fast_walker.json'


class Walker2DPDRefEnvDM(Walker2DRefEnvDM):
    def __init__(self, robot=Walker2DPD()):
        super().__init__(robot=robot)


class FixedWalkerRefEnvDM(Walker2DRefEnvDM):
    r_names = ['jpos', 'jvel']
    def __init__(self, robot=FixedWalker()):
        super().__init__(robot=robot)


class FixedWalker2DPDRefEnvDM(FixedWalkerRefEnvDM):
    def __init__(self):
        super().__init__(robot=FixedPDWalker())


if __name__ == '__main__':
    # env = FastWalker2DRefEnvDM()
    # env = Walker2DRefEnvDM()
    # env.render('human')
    # env.play_path()
    env = FixedWalker2DPDRefEnvDM()
    env.render('human')
    import time
    env._reset()
    env._p.setGravity(0.0,0.0,0.0)
    for i in range(1000):
        env._step(env.ref.pose_at_time(env.timer)[3:])
        time.sleep(env.scene.dt)
