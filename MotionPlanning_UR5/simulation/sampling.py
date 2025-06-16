import copy
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation

from simulation.environment import get_bin_pose_4x4, Environment, patch_pybullet_planning_client
from simulation.robots import get_ik_solution


def sample_goal_tcp_pose(dims, location, random_state):
    # in (0, 1)
    random_xyz = np.random.uniform(size=(3,))

    offset = np.array([0.5, 0.5, 0.0])
    random_xyz = (random_xyz - offset)
    # in (-0.5, 0.5) for first two and (0, 1) for last

    random_xyz = random_xyz * dims
    # in (-1/2 bin width, 1/2 bin width)
    # in (-1/2 bin height, 1/2 bin height)
    # in (0, bin_depth),

    bin_transform = get_bin_pose_4x4(location)
    bin_transform = bin_transform[:3]

    random_xyz_homogeneous = np.ones((4,))
    random_xyz_homogeneous[:3] = random_xyz

    random_xyz_in_bin = bin_transform @ random_xyz_homogeneous

    # x, y, z, w as pybullet wants it
    random_rotation = Rotation.random(random_state=random_state).as_quat()

    return random_xyz_in_bin, random_rotation


def sample_init(collision_fn):
    custom_limits = [
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    ]
    while True:
        init_conf = []
        for i in range(6):
            init_conf.append(np.random.uniform(custom_limits[i][0],custom_limits[i][1]))
        
        if not collision_fn(init_conf):
            return init_conf


def generate_pick_task(job):
    cfg, init, dims, location, random_state = job
    env = Environment(cfg, False)

    while True:
        goal_xyz, goal_orientation = sample_goal_tcp_pose(dims, location, random_state)

        env.set_joint_values(init)

        with patch_pybullet_planning_client(env.p._client):
            configuration_pick = get_ik_solution(
                env,
                goal_xyz,
                goal_orientation)

        if configuration_pick is None:
            continue

        return init, configuration_pick


def generate_place_task(job):
    cfg, init, place, dims, location, random_state = job
    env = Environment(cfg, False)

    while True:
        goal_xyz, goal_orientation = sample_goal_tcp_pose(dims, location, random_state)

        env.set_joint_values(init)

        configuration_pick = get_ik_solution(
            env,
            goal_xyz,
            goal_orientation)

        if configuration_pick is None:
            continue

        return configuration_pick, place


class TaskGenerator:
    def __init__(self, random_seed, task_function, pool_size=8):
        self.random_state = np.random.default_rng(seed=random_seed)
        self.task_function = task_function
        self.pool_size = pool_size

    def get_random_state(self):
        # Move the generator forward so each job has a different random state
        self.random_state.random(1000)
        return copy.deepcopy(self.random_state)

    def __next__(self):
        return self.task_function(self._get_job())

    def _get_job(self):
        raise NotImplementedError()

    def parallel_generate(self, n):
        pool = Pool(self.pool_size)
        jobs = [self._get_job() for _ in range(n)]
        return pool.map(self.task_function, jobs)


class PickTaskGenerator(TaskGenerator):
    def __init__(
            self,
            cfg,
            initial_configuration_rads,
            dims,
            location,
            random_seed,
            pool_size=8):
        super().__init__(random_seed, generate_pick_task, pool_size)
        self.cfg = cfg
        self.initial_configuration_rads = initial_configuration_rads
        self.dims = dims
        self.location = location

    def _get_job(self):
        return (
            self.cfg,
            self.initial_configuration_rads,
            self.dims,
            self.location,
            self.get_random_state()
        )


class PlaceTaskGenerator(TaskGenerator):
    def __init__(
            self,
            cfg,
            initial_configuration_rads,
            place_configuration_rads,
            dims,
            location,
            random_seed,
            pool_size=8):
        super().__init__(random_seed, generate_place_task, pool_size)
        self.cfg = cfg
        self.initial_configuration_rads = initial_configuration_rads
        self.place_configuration_rads = place_configuration_rads
        self.dims = dims
        self.location = location
        self.random_state = np.random.RandomState(random_seed)

    def _get_job(self):
        return (
            self.cfg,
            self.initial_configuration_rads,
            self.place_configuration_rads,
            self.dims,
            self.location,
            self.get_random_state()
        )

