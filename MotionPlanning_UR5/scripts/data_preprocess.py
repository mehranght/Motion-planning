import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch import nn
import pybullet as p
import pybullet_data

from pybullet_planning import connect, disconnect, \
    get_movable_joints, get_links, set_joint_positions, \
    wait_if_gui, plan_joint_motion1, plan_joint_motion, get_collision_fn
from simulation.robots import get_robot, get_ik_solution
import yaml

from simulation.environment import get_obstacles
from simulation.planning import plan_path
from simulation.robots import get_robot


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
    parser.add_argument('--data_train', default='train_place_bp.pt')
    parser.add_argument('--data_val', default='val_place_bp.pt')
    return parser.parse_args()


def resampling(path,dis_step):
    new_path = []
    for i in range(len(path)-1):
        remains = path[i+1]-path[i]
        distance = torch.norm(remains)
        num_segments = distance / dis_step
        num_segments = int(torch.ceil(num_segments).item())
        steps = torch.linspace(0, 1, num_segments)
        for j in steps:
            new_path.append(path[i] + j*remains)
    final_path = torch.zeros((len(new_path),6))
    for i in range(len(new_path)):
        final_path[i,:] = new_path[i]


    return final_path


def main(config,data_train,data_val):
    train_data = torch.load(data_train)
    val_data = torch.load(data_val)
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    connect(use_gui=False)
    obstacles = get_obstacles(**cfg['bin'])
    robot = get_robot()
    joints = get_movable_joints(robot)
    collision_fn = get_collision_fn(
        robot,
        joints,
        obstacles)
    
    train_data = train_data['paths']
    val_data = val_data['paths']


    train_data_p = []
    val_data_p = []
    train_path = []
    val_path = []
    for i in range(len(train_data)):
        train_data_p.append(LSC(train_data[i],collision_fn))
        train_path.append(resampling(train_data_p[i], 10*np.pi/180 ))
        print('Num so far: train', i)

    for i in range(len(val_data)):
        val_data_p.append(LSC(val_data[i],collision_fn))
        val_path.append(resampling(val_data_p[i], 10*np.pi/180 ))
        print('Num so far: val', i)




    torch.save(
        {
            'paths': train_path
        },
        'train_data_place.pt')

    torch.save(
        {
            'paths': val_path
        },
        'val_data_place.pt')



    

    


if __name__ == '__main__':
    parsed_args = parse_args()
    main(**vars(parsed_args))