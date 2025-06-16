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
    parser.add_argument('--weights', default='5999.pt')
    return parser.parse_args()

def main(config, weights):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    connect(use_gui=False)
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

    model = get_baxter_mlp(12, 6, dropout_rate= 0.0)
    model.load_state_dict(torch.load(weights))
    # model.eval()

    length_model = []
    length_oracle = []
    time_model = []
    time_oracle = []
    model_failure = 0
    oracle_failure = 0
    n_traj = 100
    for j in range(n_traj):
        
        ik_t1 = time.time()
        while True:
            init_xyz, init_orientation = sample_goal_tcp_pose(**cfg['bin'])

            set_joint_positions(robot, joints, goal_configuration_rads)

            init_configuration = get_ik_solution(
                robot,
                joints,
                init_xyz,
                init_orientation,
                collision_fn)
            # print('Goal pos is:',goal_xyz)
            # print('Goal orientation is:',goal_orientation)
            if init_configuration is not None:
                break
        ik_t2 = time.time()
        # print('IK time', ik_t2-ik_t1)
        plan_oracle = False
        init_configuration = torch.FloatTensor(init_configuration)
        goal_configuration = torch.FloatTensor(goal_configuration_rads)
        tic = time.time()
        path_model = MPnetPath(init_configuration,goal_configuration,model,collision_fn,0.05,plan_oracle,robot,obstacles,joints,CUSTOM_LIMITS)
        toc = time.time()
        
        tic1 = time.time()
        path_oracle = plan_path_place1(
            robot,
            joints,
            collision_fn,
            CUSTOM_LIMITS,
            goal_configuration_rads,
            obstacles,
            init_configuration)
        toc1 = time.time()
        if path_model is None:
            print('Model Fail')
            model_failure += 1
            # disconnect()
        if path_oracle is None:
            print('Oracle Fail')
            oracle_failure += 1
            # disconnect()
        
        if path_model and path_oracle is not None:
            print('Time',toc - tic)
            time_model.append(toc - tic)
            time_oracle.append(toc1 - tic1)
            path_model = [points.numpy() for points in path_model]


            
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