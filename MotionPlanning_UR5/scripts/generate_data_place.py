import argparse

import numpy as np
import torch
import yaml

from simulation.sampling import sample_goal_tcp_pose
from simulation.planning import PlanerBase

CUSTOM_LIMITS = [
    (-2*np.pi, 2*np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
]


def parse_args():
    parser = argparse.ArgumentParser()
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
        default='out_place.pt',
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

    goal_configuration_rads = [
        (np.pi / 180) * x for x in cfg['robot']['goal_configuration_degs']]

    paths = []
    goal_poses = []

    while True:
        init_xyz, init_orientation = sample_goal_tcp_pose(
            **cfg['bin'])

        path = sim.plan_path_from_xyz_orientation_to_configuration(
            goal_configuration_rads,
            init_xyz,
            init_orientation)

        if path is not None:
            goal_poses.append(
                (torch.FloatTensor(init_xyz),
                 torch.FloatTensor(init_orientation)))
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
