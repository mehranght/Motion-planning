import argparse

import torch
import yaml

from simulation.sampling import sample_goal_tcp_pose, sample_init
from simulation.planning import PlanerBase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='basic.yaml',
        type=str,
        help="")  # TODO: writeme
    parser.add_argument(
        '--num-trajectories',
        required=True,
        type=int,
        help="")  # TODO: writeme
    parser.add_argument(
        '--output',
        default='out_rand_init_place.pt',
        help="")  # TODO: writeme
    parser.add_argument(
        '--visualize',
        action='store_true',
        default= False,
        help="")  # TODO: writeme
    return parser.parse_args()


def main(config, num_trajectories, output, visualize):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    sim = PlanerBase(cfg, visualize)

    paths = []
    goal_poses = []

    while True:
        goal_xyz, goal_orientation = sample_goal_tcp_pose(
            **cfg['bin'])
        initial_configuration_rads = sample_init(sim.collision_fn)

        path = sim.plan_path_to_xyz_orientation(
            initial_configuration_rads,
            goal_xyz,
            goal_orientation)

        if path is not None:
            goal_poses.append(
                (torch.FloatTensor(goal_xyz),
                 torch.FloatTensor(goal_orientation)))
            paths.append(torch.FloatTensor(path))

            num_so_far = len(paths)
            print(num_so_far)
            if num_so_far >= num_trajectories:
                break

    torch.save(
        {
            'goal_poses': goal_poses,
            'paths': paths
        },
        output)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(**vars(parsed_args))
