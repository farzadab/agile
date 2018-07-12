import numpy as np

from .robots import Walker2D, Walker2DNoMass
from .modified_base_envs import WalkerBaseBulletEnv
from .walker_paths import WalkingPath


class Walker2DEnv(WalkerBaseBulletEnv):
    def __init__(self):
        self.robot = Walker2D()
        WalkerBaseBulletEnv.__init__(self, self.robot)


class Walker2DRefEnv(Walker2DEnv):
    def __init__(self, rsi=True):
        '''
        @param rsi: whether or not to do Random-Start-Initialization (default: True)
        '''
        self.timer = 0
        self.rsi = rsi
        self.ref_robot = Walker2DNoMass()
        super().__init__()
        self.ref = WalkingPath
        self.correct_state_id = False

    def play_path(self, timesteps=1000):
        'Replays the reference trajectory'
        import time
        self._reset()
        for i in range(timesteps):
            pose = self.ref.pose_at_time(self.timer)
            self.reset_ref_pose(pose)
            self.robot.body_xyz = pose[:3]
            # self.camera_adjust(2)
            self.timer += self.scene.dt
            time.sleep(self.scene.dt)
            # if (i+1) % 20 == 0:
            self.scene.global_step()

    def _seed(self, seed=None):
        ret = super()._seed(seed)
        self.ref_robot.np_random = self.np_random
        return ret
    
    def reset_ref_pose(self, pose):
        if self.isRender:
            pose[1] = 1
            self.ref_robot.reset_stationary_pose(pose[:3], pose[3:])

    def reset_stationary_pose(self, pose):
        self.robot.reset_stationary_pose(pose[:3], pose[3:])
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
        self.reset_stationary_pose(pose)

        return self.robot.calc_state()

    def _step(self, action):
        self.timer += self.scene.dt
        obs, rew, done, extra = super()._step(action)
        pose = self.robot.get_stationary_pose()

        ref_pose = self.ref.pose_at_time(self.timer)
        self.reset_ref_pose(ref_pose)
        weights = np.array([15, 0, 2, 1, 1, 1, 1, 1, 1])  # mean actually gives us: [10/9, 0/9, 2/9 ....]
        diff = np.exp(-1 * (ref_pose - pose) ** 2)
        ref_reward = np.mean(weights * diff)

        rew += ref_reward

        rewards = extra.get('rewards', {})
        rewards['ref_pose'] = ref_reward
        extra['rewards'] = rewards

        rew += -1 * rewards['progress']
        rewards['progress'] *= 3

        return obs, rew, done, extra


if __name__ == '__main__':
    env = Walker2DRefEnv()
    env.render('human')
    env.play_path()
