'''
Simple kinematic paths to follow
'''
import numpy as np

class PhasePath(object):
    'Abstract class that defines a kinematic path'
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
    def __init__(self, points, seconds_per_point=1, closed=True):
        if closed:
            points.append(points[0])
        self.points = np.array(points)
        self.num_segments = len(self.points)-1
        self.seconds_per_point = seconds_per_point
    def duration(self):
        return self.num_segments * self.seconds_per_point
    def at_point(self, phase):
        print(phase)
        seg_index = min(int(phase * self.num_segments), self.num_segments-1)
        pa = self.points[seg_index]
        pb = self.points[seg_index+1]
        seg_phase = phase * self.num_segments - seg_index
        return seg_phase * pb + (1-seg_phase) * pa


class LineBFPath(DiscretePath):
    'Back and forth on a line'
    def __init__(self, length=2):
        super().__init__(
            [
                [-length / 4, 0],
                [+length / 4, 0],
            ],
            seconds_per_point=10.0,
            closed=True,
        )
