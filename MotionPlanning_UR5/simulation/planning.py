import math

import numpy as np
from pybullet_planning import wait_if_gui
import time
import torch

from simulation.environment import Environment, patch_pybullet_planning_client
from simulation.pb_planning import plan_joint_motion
from simulation.robots import get_ik_solution


def plan_with_generator(
        task_generator,
        planner,
        num_trajectories):

    data = {
        'initial': [],
        'goal': [],
        'path': [],
        'success': [],
        'time': [],
        'checks': []
    }

    if isinstance(task_generator, list):
        task_generator = iter(task_generator)

    while True:
        try:
            initial, goal = next(task_generator)
        except StopIteration:
            break
        start = time.perf_counter()
        success, path, collision_checks = \
            planner.plan_between_configurations(initial, goal)
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
        data['checks'].append(collision_checks)

        num_so_far = len(data['path'])
        if num_so_far >= num_trajectories:
            break

    data['initial'] = torch.stack(data['initial'])
    data['goal'] = torch.stack(data['goal'])
    data['success'] = torch.tensor(data['success'])
    data['time'] = torch.tensor(data['time'])
    data['checks'] = torch.tensor(data['checks'])

    return data


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


class PlanerBase:
    def __init__(self, env: Environment, discretization_step, debug=False):
        self.env = env
        self.discretization_step = discretization_step
        self.debug = debug

    def plan_path_to_tcp_pose(
            self,
            initial_configuration_rads,
            goal_xyz,
            goal_orientation):

        self.env.set_joint_values(initial_configuration_rads)

        goal_configuration = get_ik_solution(
            self.env,
            goal_xyz,
            goal_orientation)

        if goal_configuration is None:
            return None, None

        success, path, collision_checks = self.plan_between_configurations(
            initial_configuration_rads, goal_configuration)

        if path is not None:
            if self.env.visualize:
                wait_if_gui()

        return goal_configuration, path

    def plan_between_configurations(
            self,
            init_configuration_rads,
            goal_configuration_rads):
        success, path, collision_checks = self._plan_between_configurations(
            init_configuration_rads, goal_configuration_rads)
        if success and self.debug:
            assert self.is_feasible(path)
        return success, path, collision_checks

    def _plan_between_configurations(
            self,
            init_configuration_rads,
            goal_configuration_rads):
        raise NotImplementedError

    def steer_to(self, start, end):
        """ My own implementatio of this horrible code
        https://github.com/anthonysimeonov/baxter_mpnet_experiments/blob/55a1cac928874bf41db04f71bc8419f021c55b09/neuralplanner_functions.py#L13

        :param start:
        :param end:
        :return:
        """
        remains = end - start
        distance = torch.norm(remains)

        if distance > 0:
            num_segments = distance / self.discretization_step
            num_segments = int(torch.ceil(num_segments).item())
            steps = torch.linspace(0, 1, num_segments)

            for step in steps:
                state = start + step * remains
                if self.env.collision_fn(state):
                    return False

        return True

    def is_feasible(self, path):
        for i in range(len(path)-1):
            if not self.steer_to(path[i], path[i+1]):
                return False
        return True

    def binary_state_contraction(self, path):
        if len(path) <= 2:
            # Can't contract a path that's already minimum
            return path
        if self.steer_to(path[0], path[-1]):
            # We can connect the start to the end
            return torch.stack([path[0], path[-1]], dim=0)
        if len(path) == 3:
            # We could not connect the start to the end, and since its length 3
            # we can't contract any further
            return path

        while True:
            if len(path) % 2 == 1:
                # Path is odd length at least 5
                half = len(path) / 2
                half_down = int(math.floor(half))
                # length 5 would be 2.5 would be 2
                left = self.binary_state_contraction(path[:(half_down + 1)])
                right = self.binary_state_contraction(path[half_down:])
                # since path is odd length the end of left is the start of right
                new = torch.cat((left[:-1], right), dim=0)
            else:
                # Path is even and at least 4
                half = int(len(path) / 2)
                # length 4 would be 2
                # try to contract states 0 to 2 if that worked, add on state 3
                # if that failed, try to contract states 1, 2, 3 and tack on 0
                left = self.binary_state_contraction(path[:(half + 1)])
                if len(left) < half + 1:
                    # contraction happened
                    right = path[(half + 1):]
                else:
                    left = path[:(half - 1)]
                    right = self.binary_state_contraction(path[(half - 1):])
                new = torch.cat((left, right), dim=0)

            if len(new) == len(path):
                return new
            else:
                path = new

    def lazy_sate_contraction(self, path):
        for i in range(len(path) - 1):
            for j in range(len(path) - 1, i + 1, -1):
                if self.steer_to(path[i], path[j]):
                    pc = []
                    for k in range(i+1):
                        pc.append(path[k])
                    for k in range(j,len(path)):
                        pc.append(path[k])
                    return self.lazy_sate_contraction(pc)
        return path


class BiRRTPlanner(PlanerBase):
    def __init__(
            self,
            do_lsc,
            do_binary_contraction,
            do_resample,
            resample_step_size,
            *args,
            **kwargs):

        super().__init__(*args, **kwargs)
        self.do_lsc = do_lsc
        self.do_binary_contraction = do_binary_contraction
        self.do_resample = do_resample
        self.resample_step_size = resample_step_size

    def _plan_between_configurations(
            self,
            init_configuration_rads,
            goal_configuration_rads):

        self.env.set_joint_values(init_configuration_rads)

        with patch_pybullet_planning_client(self.env.p._client):
            path, collision_checks = plan_joint_motion(
                self.env.robot,
                self.env.joints,
                goal_configuration_rads,
                obstacles=self.env.obstacles,
                self_collisions=True,
                custom_limits=self.env.custom_limits,
                resolutions=[np.pi/360 for _ in range(6)],
                diagnosis=True,
                coarse_waypoints=False)

        self.env.set_joint_values(goal_configuration_rads)

        success = path is not None

        if success:
            path = torch.tensor(path, dtype=torch.float32)

        if success and self.do_lsc:
            path = self.lazy_sate_contraction(path)

        if success and self.do_binary_contraction:
            path = self.binary_state_contraction(path)

        if success and self.do_resample:
            path = resampling(path, self.resample_step_size)

        return success, path, collision_checks


def do_birrt_job(job):
    cfg, ds, init, goal = job
    env = Environment(cfg, False)
    planner = BiRRTPlanner(env=env, discretization_step=ds, **cfg['oracle'])
    out = plan_with_generator([(init, goal)], planner, 1)
    del planner
    del env.p
    del env
    return out


class NNPlanner(PlanerBase):
    def __init__(
            self,
            env: Environment,
            discretization_step,
            model: torch.nn.Module,
            oracle: BiRRTPlanner,
            plan_oracle,
            do_postprocessing,
            max_steps=100,
            **kwargs):
        super().__init__(env, discretization_step, **kwargs)
        self.oracle = oracle
        self.model = model
        self.plan_oracle = plan_oracle
        self.do_postprocessing = do_postprocessing
        self.max_steps = max_steps

    # based on Algorithm 5 in MPnet paper
    # mono_planner
    def _plan_between_configurations(
            self,
            init_configuration_rads,
            goal_configuration_rads):
        self.env.collision_check_counter = 0

        goal_configuration = torch.FloatTensor(goal_configuration_rads)
        current_configuration = torch.FloatTensor(init_configuration_rads)
        path = [current_configuration]
        step_idx = 0
        while True and step_idx < self.max_steps:
            if self.steer_to(current_configuration, goal_configuration):
                path.append(goal_configuration)
                path = torch.stack(path)
                if self.do_postprocessing:
                    path = self.binary_state_contraction(path)
                return True, path, self.env.collision_check_counter

            model_input = torch.cat(
                (current_configuration, goal_configuration))

            while True:
                step_idx += 1
                if step_idx > self.max_steps:
                    break
                with torch.no_grad():
                    model_output = self.model(model_input.unsqueeze(0))
                model_output = model_output.squeeze(0)

                if self.env.collision_fn(model_output):
                    continue

                if self.steer_to(current_configuration, model_output):
                    path.append(model_output)
                    break

                if self.plan_oracle:
                    success_prime, path_prime, _ = \
                        self.oracle.plan_between_configurations(
                            current_configuration, model_output)
                    if path_prime is None:
                        break
                    else:
                        path_prime = [
                            torch.tensor(points, dtype=torch.float32) for
                            points in path_prime]
                        path.extend(path_prime)

            current_configuration = model_output

        # Did not hit return path above because we failed, so return None
        return False, path, self.env.collision_check_counter


class BiNNPlanner(NNPlanner):
    def __init__(self, *args, iteration, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration = iteration

    def plan_between_configurations(
            self,
            init_configuration_rads,
            goal_configuration_rads):

        path_a = [init_configuration_rads]
        path_b = [goal_configuration_rads]
        path = []
        step = 0
        while step < self.iteration:
            model_input = torch.cat((path_a[-1], path_b[-1]))
            with torch.no_grad():
                model_output = self.model(model_input.unsqueeze(0)).squeeze(0)
            path_a.append(model_output)
            connect = self.steer_to(path_a[-1], path_b[-1])

            if connect:
                path.extend(path_a)
                path.extend(path_b)
                return True, path

            path_a, path_b = path_b, path_a
            step += 1

        return False, None
