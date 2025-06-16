import argparse
import time
from collections import defaultdict
import logging
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.tensorboard
from torch.utils.data.dataloader import DataLoader
import numpy as np
import yaml

from logger import set_logging_config
from ml.dataloader import MPDataSet
from ml.models import get_baxter_mlp
from simulation.environment import Environment
from simulation.planning import BiRRTPlanner, NNPlanner, do_birrt_job, plan_with_generator
from simulation.sampling import PickTaskGenerator, PlaceTaskGenerator
from scripts.dagger import degs_to_rads

import os
os.chdir('MotionPlanning_UR5')

with open('configs/basic.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

initial_configuration_rads = degs_to_rads(
    cfg['robot']['initial_configuration_degs'])

task_generator = PickTaskGenerator(
        cfg,
        initial_configuration_rads,
        **cfg['bin'],
        random_seed=0,
        pool_size=4)

asdf = task_generator.parallel_generate(4)
for a in asdf:
    print(a[1])