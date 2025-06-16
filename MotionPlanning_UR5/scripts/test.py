import argparse

import torch
from pybullet_planning import wait_if_gui
import yaml

from ml.models import get_baxter_mlp
from simulation.environment import Environment
from simulation.sampling import PickTaskGenerator, PlaceTaskGenerator
from simulation.planning import NNPlanner, BiRRTPlanner
from scripts.dagger import discretization_step, degs_to_rads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/basic.yaml',
        type=str,
        help="")  # TODO: writeme
    parser.add_argument('--weights', default='5999.pt')
    parser.add_argument('--which', choices=['pick', 'place'], default='pick')
    return parser.parse_args()


def main(config, weights, which):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    env = Environment(cfg, True)
    oracle = BiRRTPlanner(
        env=env,
        discretization_step=discretization_step,
        **cfg['oracle'])
    model = get_baxter_mlp(12, 6, cfg['model']['dropout'])
    model.load_state_dict(torch.load(weights))
    nn_planner = NNPlanner(env, discretization_step, model, oracle, False, True)

    initial_configuration_rads = degs_to_rads(
        cfg['robot']['initial_configuration_degs'])
    place_configuration_rads = degs_to_rads(
        cfg['robot']['goal_configuration_degs'])

    if which == 'pick':
        task_generator = PickTaskGenerator(
            cfg,
            initial_configuration_rads,
            random_seed=42,
            **cfg['bin'])
    else:
        task_generator = PlaceTaskGenerator(
            cfg,
            initial_configuration_rads,
            place_configuration_rads,
            random_seed=42,
            **cfg['bin'])

    while True:
        init, goal = next(task_generator)
        success, path = nn_planner.plan_between_configurations(init, goal)

        if not success:
            print('Fail')

        if path is None:
            continue

        path = [points.numpy() for points in path]
        for i, conf in enumerate(path):
            conf = conf.reshape(6)
            env.set_joint_values(conf)
            residual = goal - conf
            print(f'Difference to goal: {residual}')
            wait_if_gui('Step: {}/{}'.format(i, len(path)))
        print('EE position', env.get_link_state(9))
        print(path)
        wait_if_gui('Finish?')


if __name__ == '__main__':
    parsed_args = parse_args()
    main(**vars(parsed_args))

