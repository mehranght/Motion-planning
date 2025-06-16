import argparse
from importlib.resources import path
import time
import numpy as np
import pybullet as p
import torch
from pybullet_planning import (
    connect,
    disconnect,
    get_collision_fn,
    get_movable_joints,
    set_joint_positions,
    wait_if_gui)
import yaml
from ml.models import get_baxter_mlp
from simulation.environment import get_obstacles
from simulation.robots import get_robot, get_ik_solution
from simulation.sampling import sample_goal_tcp_pose
from simulation.planning import plan_path, plan_path1
from generate_data import CUSTOM_LIMITS
from test import (
    steer_to,
    BNP,
    mono_planner,
    replan,
    LSC,
    isFeasible,
    MPnetPath)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='basic.yaml',
        type=str,
        help="")  # TODO: writeme
    parser.add_argument('--weights', default='final_place_dropout.pt')
    return parser.parse_args()

def main(config, weights):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    connect(use_gui=True)
    p.setAdditionalSearchPath('simulation/')
    p.setAdditionalSearchPath('simulation/table/')

    obstacles = get_obstacles(**cfg['bin'])
    goal_configuration_rads = [
        (np.pi / 180) * x for x in cfg['robot']['goal_configuration_degs']]

    robot = get_robot()
    joints = get_movable_joints(robot)
    collision_fn = get_collision_fn(
        robot,
        joints,
        obstacles)

    model = get_baxter_mlp(12, 6, 0.0)
    model.load_state_dict(torch.load(weights))

    while True:
        init_xyz, init_orientation = sample_goal_tcp_pose(**cfg['bin'])

        set_joint_positions(robot, joints, goal_configuration_rads)

        init_configuration = get_ik_solution(
            robot,
            joints,
            init_xyz,
            init_orientation,
            collision_fn)
        print('Goal pos is:',init_xyz)
        print('Goal orientation is:',init_orientation)
        if init_configuration is not None:
            break

    plan_oracle = False
    init_configuration = torch.FloatTensor(init_configuration)
    goal_configuration = torch.FloatTensor(goal_configuration_rads)
    path = MPnetPath(init_configuration,goal_configuration,model,collision_fn,0.05,plan_oracle,robot,obstacles,joints,CUSTOM_LIMITS)
    if path is None:
        print('Fail')
        disconnect()
    path = [points.numpy() for points in path]
    for i, conf in enumerate(path):
        set_joint_positions(robot, joints, conf.reshape(6))
        wait_if_gui('Step: {}/{}'.format(i, len(path)))
    print('EE position',p.getLinkState(robot, 9))
    print(path)
    wait_if_gui('Finish?')
    disconnect()


if __name__ == '__main__':
    parsed_args = parse_args()
    main(**vars(parsed_args))