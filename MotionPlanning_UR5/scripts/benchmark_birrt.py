import time

import yaml



with open('configs/basic.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

from simulation.environment import Environment
from simulation.planning import BiRRTPlanner
from simulation.sampling import PickTaskGenerator
from scripts.dagger import discretization_step, degs_to_rads


initial_configuration_rads = degs_to_rads(
    cfg['robot']['initial_configuration_degs'])
place_configuration_rads = degs_to_rads(
    cfg['robot']['goal_configuration_degs'])

task_generator = PickTaskGenerator(
    cfg,
    initial_configuration_rads,
    **cfg['bin'])
env = Environment(cfg, False)
oracle = BiRRTPlanner(
            env=env,
            discretization_step=discretization_step,
            **cfg['oracle'])


n = 10

times = []
for _ in range(n):
    t = next(task_generator)
    start = time.perf_counter()
    out = oracle.plan_between_configurations(*t)
    end = time.perf_counter()
    times.append(end - start)
print(times)
