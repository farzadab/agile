'''
Simple kinematic paths to follow
'''
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

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

    def one_cycle_duration(self):
        return self.duration

    def pose_at_time(self, time):
        pose = np.zeros(self.periodic.shape)
        pose[self.periodic] = self.cyclic_sp(time)
        cycle_num = np.floor(time / self.duration)
        phase = time - cycle_num * self.duration
        pose[np.invert(self.periodic)] = self.acyclic_sp(phase) + cycle_num * self.acyclic_diff
        return pose


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