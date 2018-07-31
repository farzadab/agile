'''
Module for reading/importing motion capture data
For now we only use motions from `TerrainRLSim` project: https://github.com/UBCMOCCA/TerrainRLSim/tree/master/data/motions
A description of the motion files is available here (private repo): https://github.com/UBCMOCCA/TerrainRL/wiki/Motion-File#3d-example
'''
import os
import json
import numpy as np
from pyquaternion import Quaternion

from logs import logger
from .paths import RepeatingPath

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MOCAP_PATH = os.path.join(CUR_DIR, 'ref_data')

def _get_orientation_from_quaternion(elements):
    q = Quaternion(elements)
    return q.get_axis()[2] * q.angle

def _get_orientations_from_quaternion_seq(seq):
    return [_get_orientation_from_quaternion(v) for v in seq]

def importTRLmotion3to2(filename, path=MOCAP_PATH):
    '''
    Imports motion data from TerrainRLSim motion files.
    It assumes that the character type is the `biped3d.txt`.
    '''
    file_path = os.path.join(path, filename)
    with open(file_path, 'r') as fp:
        data = json.load(fp)
        if 'Loop' in data and data['Loop'] is not True:
            logger.warning('This data was not defined to loop around')
        if 'Frames' not in data:
            raise ValueError(
                'The file `%s` does not conform to the correct TRL MoCap format' % file_path
            )

        frames = np.array(data['Frames'])
        duration = frames[:-1, 0].sum()

        torso = frames[:, 1:4]

        torso[:, 2] += data.get('ZCompensation', 1.0)
        # torso[:, 1] += data.get('YCompensation', -0.76)
        torso[:, 1] = 0
        # skipping the root and torso orientation (always assuming upright position for now)
        joints = frames[:, 12:]

        joints_2d = np.vstack([
            _get_orientations_from_quaternion_seq(joints[:, :4]),
            joints[:, 4],
            _get_orientations_from_quaternion_seq(joints[:, 5:9]),
            _get_orientations_from_quaternion_seq(joints[:, 9:13]),
            joints[:, 13],
            _get_orientations_from_quaternion_seq(joints[:, 14:18]),
        ]).transpose()

        points = np.hstack([torso, joints_2d])
        periodic = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool)

        print('error: %.4f' % ((points[-1, periodic] - points[0, periodic]) ** 2).sum())
        points[-1, periodic] = points[0, periodic]

        return RepeatingPath(
            duration=duration, # * 10
            points=points,
            periodic=periodic,
        )
