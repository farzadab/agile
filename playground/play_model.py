# Just a simple environment tester, based on https://github.com/FracturedPlane/RLSimulationEnvironments/blob/master/EnvTester.py
import numpy as np
import argparse
import ipdb
import time
import pybullet as p
import pybullet_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Just displaying a model in PyBullet"
    )
    parser.add_argument(
        "--filename", help="address of the model to use"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print('loading %s' % args.filename)

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    modelId = p.loadMJCF(args.filename) #[0, 0, 0], p.getQuaternionFromEuler([0, 0, 0])

    for i in range(10000):
        p.stepSimulation()
        time.sleep(1/240)
    p.disconnect()

if __name__ == '__main__':
    main()
