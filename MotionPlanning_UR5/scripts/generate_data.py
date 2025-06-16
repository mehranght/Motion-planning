import argparse
import time

import numpy as np
import torch
import yaml

from ml.models import get_baxter_mlp
from simulation.environment import Environment
from simulation.robots import get_ik_solution, set_joint_positions
from simulation.sampling import sample_goal_tcp_pose, bin_tcp_pose_generator
from simulation.planning import BiRRTPlanner, NNPlanner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', choices=['pick', 'place'])
    parser.add_argument(
        '--config',
        default='configs/basic.yaml',
        type=str,
        help="")  # TODO: writeme
    parser.add_argument(
        '--num-trajectories',
        required=True,
        type=int,
        help="")  # TODO: writeme
    parser.add_argument(
        '--output',
        default='out_temp.pt',
        help="")  # TODO: writeme
    parser.add_argument(
        '--visualize',
        action='store_true',
        default= False,
        help="")  # TODO: writeme
    parser.add_argument(
        '--planner-name',
        choices=['birrt', 'nn'],
        default='birrt')
    parser.add_argument('--checkpoint', required=False, type=str)
    parser.add_argument('--only-success', action='store_true')
    return parser.parse_args()


def plan_with_generator(
        task_generator,
        planner,
        num_trajectories):

    data = {
        'initial': [],
        'goal': [],
        'path': [],
        'success': [],
        'time': []
    }

    while True:
        try:
            initial, goal = next(task_generator)
        except StopIteration:
            break
        start = time.perf_counter()
        success, path = planner.plan_between_configurations(initial, goal)
        duration = time.perf_counter() - start

        if not isinstance(initial, torch.Tensor):
            initial = torch.FloatTensor(initial)
        if not isinstance(goal, torch.Tensor):
            goal = torch.FloatTensor(goal)
        if path is not None:
            if not isinstance(path, torch.Tensor):
                if isinstance(path[0], torch.Tensor):
                    path = torch.stack(path)
                else:
                    path = torch.FloatTensor(path)

        data['initial'].append(initial)
        data['goal'].append(goal)
        data['path'].append(path)
        data['success'].append(success)
        data['time'].append(duration)

        num_so_far = len(data['path'])
        print(num_so_far)
        if num_so_far >= num_trajectories:
            break

    data['initial'] = torch.stack(data['initial'])
    data['goal'] = torch.stack(data['goal'])
    data['success'] = torch.tensor(data['success'])
    data['time'] = torch.tensor(data['time'])

    return data


def main(
        which,
        config,
        num_trajectories,
        output,
        visualize,
        planner_name,
        checkpoint,
        only_success):

    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    env = Environment(cfg, visualize)
    planner = BiRRTPlanner(env, 0.05)
    if planner_name == 'nn':
        model = get_baxter_mlp(12, 6, 0.0)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)
        planner = NNPlanner(env, 0.05, model, planner, False, True)

    initial_configuration_rads = [
        (np.pi / 180) * x for x in cfg['robot']['initial_configuration_degs']]
    place_configuration_rads = [
        (np.pi / 180) * x for x in cfg['robot']['goal_configuration_degs']]

    goal_generator = bin_tcp_pose_generator(env, initial_configuration_rads, **cfg['bin'])
    data = plan_with_generator(
        goal_generator,
        planner,
        initial_configuration_rads,
        place_configuration_rads,
        which,
        num_trajectories,
        only_success)

    torch.save(data, output)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(**vars(parsed_args))
