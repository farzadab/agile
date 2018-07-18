'''
Simple kinematic paths to follow
'''
import numpy as np
import json
import os
from scipy.interpolate import CubicSpline, interp1d
from scipy.misc import derivative

class PhasePath(object):
    'Abstract class that defines a kinematic path dependent on phase'
    def duration(self):
        '''
        @returns a floating point value, determining the duration of the whole path
        '''
        raise NotImplementedError

    def at_point(self, phase):
        '''
        @returns x(ϕ): the position at time
        '''
        raise NotImplementedError

class TimePath(object):
    'Abstract class that defines a kinematic path dependent on time'
    def pose_at_time(self, time):
        '''
        @returns x(t): the position at time
        '''
        raise NotImplementedError


class CircularPath(PhasePath):
    'Circular Kinematic path'
    def __init__(self, radius=0.5, angular_speed=0.15):
        self.radius = radius
        self.angular_speed = angular_speed
    def duration(self):
        return 2 * np.pi / self.angular_speed
    def at_point(self, phase):
        return np.array([np.cos(phase*2*np.pi), np.sin(phase*2*np.pi)]) * self.radius


class DiscretePath(PhasePath):
    def __init__(self, points, seconds_per_point=10, closed=True, smooth=False):
        if closed:
            points.append(points[0])
        self.points = np.array(points)
        self.num_segments = len(self.points)-1
        self.seconds_per_point = seconds_per_point
        self.spline = None
        if smooth:
            self.spline = CubicSpline(
                [i/self.num_segments for i in range(self.num_segments+1)],
                self.points,
                axis=0,
                bc_type='periodic' if closed else 'not-a-knot',
                extrapolate='periodic',
            )
    def duration(self):
        return float(self.num_segments * self.seconds_per_point)
    def at_point(self, phase):
        if self.spline:
            return self.spline(phase)

        seg_index = min(int(phase * self.num_segments), self.num_segments-1)
        pa = self.points[seg_index]
        pb = self.points[seg_index+1]
        seg_phase = phase * self.num_segments - seg_index
        return seg_phase * pb + (1-seg_phase) * pa


class RepeatingPath(TimePath):
    def __init__(self, duration, points, periodic):
        '''
        @param duration: the duration of one cycle (going through all points)
        @param points: the set of data points
        @param periodic: a boolean vector indicating which indices are periodic
        '''
        # np.invert
        self.periodic = np.array(periodic, dtype=np.bool)
        self.duration = duration
        points = np.array(points)
        self.num_segments = points.shape[0]-1
        times = [i*self.duration/self.num_segments for i in range(self.num_segments+1)]
        self.acyclic_diff = (points[-1,:] - points[0,:])[np.invert(self.periodic)]
        self.acyclic_sp = interp1d(
            times,
            points[:, np.invert(self.periodic)],
            axis=0,
            fill_value='extrapolate',  # this shouldn't be needed
        )
        self.cyclic_sp = CubicSpline(
            times,
            points[:, self.periodic],
            axis=0,
            bc_type='periodic',
            extrapolate='periodic',
        )
        self.cyclic_sp_grad = self.cyclic_sp.derivative()
        self.acyclic_sp_grad = lambda x: derivative(self.acyclic_sp, x)

    def one_cycle_duration(self):
        return self.duration

    def pose_at_time(self, time):
        pose = np.zeros(self.periodic.shape)
        pose[self.periodic] = self.cyclic_sp(time)
        cycle_num = np.floor(time / self.duration)
        phase = time - cycle_num * self.duration
        pose[np.invert(self.periodic)] = self.acyclic_sp(phase) + cycle_num * self.acyclic_diff
        return pose

    def vel_at_time(self, time):
        vel = np.zeros(self.periodic.shape)
        vel[self.periodic] = self.cyclic_sp_grad(time)
        cycle_num = np.floor(time / self.duration)
        phase = time - cycle_num * self.duration
        vel[np.invert(self.periodic)] = self.acyclic_sp_grad(phase)
        return vel


class RefMotionStore(object):
    '''
    A class for storing/accessing reference motion data

    You need to call `interpolate` after populating this object with motion values.
    Then you can the following functions and parameters:
        - path
        - j_pos_ind
        - j_vel_ind
        - ee_pos_ind
        - com_pos_ind
    '''
    store_path = os.path.join(os.path.dirname(__file__), 'ref_data')
    def __init__(self, dt=None, joint_names=[], end_eff_names=[]):
        self.joint_names = joint_names
        self.end_eff_names = end_eff_names
        self.dt = dt       # time-step
        self.j_pos = []    # joint positions (orientations)
        self.j_vel = []    # joint velocities
        self.ee_pos = []   # end-effector positions
        self.com_pos = []  # center-of-mass position
        self.j_pos_ind, self.j_vel_ind, self.ee_pos_ind, self.com_pos_ind = None, None, None, None
        self.path = None

    def set_names(self, joint_names, end_eff_names):
        self.joint_names = joint_names
        self.end_eff_names = end_eff_names

    def record(self, j_pos, j_vel, ee_pos, com_pos):
        self.j_pos.append(np.array(j_pos).tolist())
        self.j_vel.append(np.array(j_vel).tolist())
        self.ee_pos.append(np.array(ee_pos).tolist())
        self.com_pos.append(np.array(com_pos).tolist())

    def at_time(self, timestep):
        '''
        @returns: joint_positions, joint_velocities, end_effector_positions, com_position
        '''
        ts = timestep % len(self.j_pos)
        return self.j_pos[ts], self.j_vel[ts], self.ee_pos[ts], self.com_pos[ts]
    
    @staticmethod
    def __get_ranges(sizes):
        ranges = []
        last = 0
        for size in sizes:
            ranges.append(np.arange(last, last+size))
            last += size
        return ranges
    
    def interpolate(self):
        if len(self.j_pos) == 0:
            raise AssertionError('You should store at least one item and then call interpolate')

        points = np.array([
            np.concatenate([
                self.j_pos[i],
                self.j_vel[i],
                self.ee_pos[i],
                self.com_pos[i],
            ])
            for i in range(len(self.j_pos))
        ])

        self.j_pos_ind, self.j_vel_ind, self.ee_pos_ind, self.com_pos_ind = self.__get_ranges([
            len(self.j_pos[0]),
            len(self.j_vel[0]),
            len(self.ee_pos[0]),
            len(self.com_pos[0]),
        ])

        periodic = np.zeros(points.shape[1], dtype=np.bool)
        periodic[self.j_pos_ind] = True
        periodic[self.j_vel_ind] = True

        print('error:', np.mean(np.square(points[-1, periodic] - points[0, periodic])))
        points[-1, periodic] = points[0, periodic]

        self.path = RepeatingPath(
            duration=self.dt * len(self.j_pos),
            points=points,
            periodic=periodic
        )
    
    def one_cycle_duration(self):
        if self.path is None:
            AssertionError('You can only call this method after calling `interpolate`')
        return self.path.one_cycle_duration()

    def pose_at_time(self, time):
        data = self.path.pose_at_time(time)
        return data[np.concatenate([self.com_pos_ind, self.j_pos_ind])]

    def vel_at_time(self, time):
        data = self.path.vel_at_time(time)
        return data[np.concatenate([self.com_pos_ind, self.j_pos_ind])]

    def ref_at_time(self, time):
        data = self.path.pose_at_time(time)
        v_data = self.path.vel_at_time(time)
        return {
            'jpos': data[self.j_pos_ind],
            'jvel': data[self.j_vel_ind],
            'ee': data[self.ee_pos_ind],
            'torso': data[self.com_pos_ind],
            'torso_z': data[self.com_pos_ind][2],
            'torso_v': v_data[self.com_pos_ind],
        }

    def load(self, file_name):
        full_path = os.path.join(self.store_path, file_name)
        with open(full_path, 'r') as fn:
            data = json.load(fn)
            self.joint_names = self.__dict__.update(data)
        self.interpolate()
        return self

    def store(self, file_name):
        os.makedirs(self.store_path, exist_ok=True)
        full_path = os.path.join(self.store_path, file_name)
        with open(full_path, 'w') as fn:
            json.dump({
                'joint_names': self.joint_names,
                'end_eff_names': self.end_eff_names,
                'dt': self.dt,
                'j_pos': self.j_pos,
                'j_vel': self.j_vel,
                'ee_pos': self.ee_pos,
                'com_pos': self.com_pos,
            }, fn)


class LineBFPath(DiscretePath):
    'Back and forth on a line'
    def __init__(self, length=2, closed=True, **kwargs):
        super().__init__(
            [
                [-length / 4, 0],
                [+length / 4, 0],
            ],
            seconds_per_point=10.0,
            closed=closed,
            **kwargs
        )

class RPath1(DiscretePath):
    points = [
        [-.6, -.8],
        [-.3, -.8],
        [0  , -.8],
        [+.3, -.8],
        [+.6, -.8],

        [+.8, -.6],
        [+.8, +.6],

        [+.6, +.8],
        [0  , +.8],
        [-.6, +.8],

        [-.8, 0  ],
        [-.8, 0  ],
        [-.8, 0  ],
        [-.8, 0  ],
    ]
    def __init__(self, seconds_per_point=2, **kwargs):
        super().__init__(self.points, seconds_per_point=seconds_per_point, **kwargs)