import numpy as np

from .robots import Walker2D
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
        super().__init__()
        self.ref = WalkingPath

    def play_path(self, timesteps=1000):
        'Replays the reference trajectory'
        import time
        self._reset()
        for i in range(timesteps):
            pose = self.ref.pose_at_time(self.timer)
            self.reset_stationary_pose(pose[:3], pose[3:])
            self.camera_adjust(2)
            self.timer += self.scene.dt
            time.sleep(self.scene.dt)

    def _reset(self):
        super()._reset()

        if self.rsi:
            self.timer = self.np_random.uniform(0, self.ref.one_cycle_duration())
        else:
            self.timer = 0

        pose = self.ref.pose_at_time(self.timer)
        self.reset_stationary_pose(pose[:3], pose[3:])

        return self.robot.calc_state()

    def _step(self, action):
        self.timer += self.scene.dt
        obs, rew, done, extra = super()._step(action)
        pose = self.get_stationary_pose()

        ref_pose = self.ref.pose_at_time(self.timer)
        ref_reward = np.exp(-np.mean((ref_pose - pose) ** 2))

        rew += ref_reward

        rewards = extra.get('rewards', {})
        rewards['ref_motion'] = ref_reward
        extra['rewards'] = rewards

        rew += 3 * rewards['progress']

        return obs, rew, done, extra


if __name__ == '__main__':
    env = Walker2DRefEnv()
    env.render('human')
    env.play_path()
