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
from simulation.planning import plan_path, plan_path1, plan_path_place1
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
    parser.add_argument('--weights_pick', default='5999.pt')
    parser.add_argument('--weights_place', default='4999.pt')
    return parser.parse_args()


def main(config, weights_pick, weights_place):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    connect(use_gui=False)
    p.setAdditionalSearchPath('simulation/')
    p.setAdditionalSearchPath('simulation/table/')

    length_model = []
    length_oracle = []
    time_model = []
    time_oracle = []
    model_failure = 0
    oracle_failure = 0
    n_traj = 1000

    obstacles = get_obstacles(**cfg['bin'])
    init_configuration_rads = [
        (np.pi / 180) * x for x in cfg['robot']['initial_configuration_degs']]
    goal_configuration_rads = [
        (np.pi / 180) * x for x in cfg['robot']['goal_configuration_degs']]

    robot = get_robot()
    joints = get_movable_joints(robot)
    collision_fn = get_collision_fn(
        robot,
        joints,
        obstacles)

    model_pick = get_baxter_mlp(12, 6, dropout_rate= 0.0)
    model_place = get_baxter_mlp(12, 6, dropout_rate= 0.0)
    model_pick.load_state_dict(torch.load(weights_pick))
    model_place.load_state_dict(torch.load(weights_place))


    for j in range(n_traj):

        while True:
            goal_xyz, goal_orientation = sample_goal_tcp_pose(**cfg['bin'])

            set_joint_positions(robot, joints, init_configuration_rads)

            goal_configuration = get_ik_solution(
                robot,
                joints,
                goal_xyz,
                goal_orientation,
                collision_fn)
            # print('Goal pos is:',goal_xyz)
            # print('Goal orientation is:',goal_orientation)
            if goal_configuration is not None:
                break
        plan_oracle = False
        goal_configuration = torch.FloatTensor(goal_configuration)
        current_configuration = torch.FloatTensor(init_configuration_rads)
        path_model = None
        path_oracle = None
        tic = time.time()
        path_pick = MPnetPath(current_configuration,goal_configuration,model_pick,collision_fn,0.05,plan_oracle,robot,obstacles,joints,CUSTOM_LIMITS)
        
        # print(path_pick)
        if path_pick is not None:
            path_place = MPnetPath(goal_configuration,goal_configuration_rads,model_place,collision_fn,0.05,plan_oracle,robot,obstacles,joints,CUSTOM_LIMITS)
            
            # print(path_place)
            if path_place is not None:
                
                # print('Done!')
                # print('Planning time:', toc-tic )
                path_pick = [points.numpy() for points in path_pick]
                path_place = [points.numpy() for points in path_place]
                path_model = np.concatenate((path_pick,path_place[1:]), axis = 0)
                # path_model = np.array(path_model)
                # print(path_model)
                # for i, conf in enumerate(path):
                #     set_joint_positions(robot, joints, conf.reshape(6))
                #     wait_if_gui('Step: {}/{}'.format(i, len(path)))
                # print('EE position',p.getLinkState(robot, 9))
                
                # wait_if_gui('Finish?')
                # disconnect()
        toc = time.time()
        tic1 = time.time()
        path_pick_o = plan_path1(
                robot=robot,
                joints=joints,
                collision_fn=collision_fn,
                custom_limits=CUSTOM_LIMITS,
                obstacles=obstacles,
                initial_configuration_rads=init_configuration_rads,
                goal_configuration = goal_configuration)

        if path_pick_o is not None:
            path_place_o = plan_path_place1(
                robot,
                joints,
                collision_fn,
                CUSTOM_LIMITS,
                goal_configuration_rads,
                obstacles,
                goal_configuration)
            if path_place_o is not None:
                path_oracle = np.concatenate((path_pick_o,path_place_o[1:]), axis = 0)
        toc1 = time.time()
        if path_model is None:
            print('Model Fail')
            model_failure += 1
        if path_oracle is None:
            print('Oracle Fail')
            oracle_failure += 1

        if (path_model is not None) and (path_oracle is not None):
            time_model.append(toc - tic)
            time_oracle.append(toc1 - tic1)
            temp1 = 0
            temp2 = 0
            for i in range(len(path_model)-1):
                d = np.linalg.norm(path_model[i]-path_model[i+1])
                temp1 +=d
            for i in range(len(path_oracle)-1):
                d = np.linalg.norm(np.array(path_oracle[i])-np.array(path_oracle[i+1]))
                temp2 +=d
            length_model.append(temp1)
            length_oracle.append(temp2)
        print('Num so far',j)
    print('Average model length', np.mean(length_model))
    print('Standard deviation model length', np.std(length_model))
    print('Average oracle length', np.mean(length_oracle))
    print('Standard deviation oracle length', np.std(length_oracle))
    print('Average model time', np.mean(time_model))
    print('Standard deviation model time', np.std(time_model))
    print('Average oracle time', np.mean(time_oracle))
    print('Standard deviation oracle time', np.std(time_oracle))
    print('model success rate', (n_traj-model_failure))
    print('oracle success rate', (n_traj-oracle_failure))


if __name__ == '__main__':
    parsed_args = parse_args()
    main(**vars(parsed_args))

