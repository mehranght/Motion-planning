"""
Train initial predictor pi0
1. Collect demonstrations from expert
2. Treat the demonstrations as iid state-action pairs
3. Learn a policy to match

For m = 1:
  - Collect tau by rolling out pi_m_prev
  - Estimate state distribution pm using s e tau
  - Collect interactive feedback from expert
  - train pi_m
"""
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


device = 'cuda:0'

learning_rate = 0.0001
save_every_n_epochs = 100
validate_every_n_epochs = 10
discretization_step = 0.05
initial_demonstrations = 500
rollouts = 100
num_states_per_iter = 500
batch_size = 128
training_epochs = 500
test_tasks = 100
dagger_iters = 10
n_proc = 10

# debug settings
# learning_rate = 0.0001
# save_every_n_epochs = 10
# validate_every_n_epochs = 10
# discretization_step = 0.05
# initial_demonstrations = 20
# rollouts = 10
# num_states_per_iter = 10
# batch_size = 128
# training_epochs = 10
# test_tasks = 2
# dagger_iters = 2


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='a')
    parser.add_argument('--config', type=str, default='configs/basic.yaml')
    parser.add_argument('--which', choices=['pick', 'place'], default='pick')
    return parser.parse_args()


def degs_to_rads(configuration):
    return [(np.pi / 180) * x for x in configuration]


class DaggerTrainer:
    def __init__(self, config, output_folder, which):
        with open(config) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        self.cfg = cfg

        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

        set_logging_config(self.output_folder / 'log.txt', True)
        self.logger = logging.getLogger()

        self.env = Environment(cfg, False)
        self.oracle = BiRRTPlanner(
            env=self.env,
            discretization_step=discretization_step,
            **cfg['oracle'])
        model = get_baxter_mlp(12, 6, cfg['model']['dropout'])
        self.nn_planner_with_postprocessing = NNPlanner(
            self.env, discretization_step, model, self.oracle, False, True)
        self.nn_planner_without_postprocessing = NNPlanner(
            self.env, discretization_step, model, self.oracle, False, False)

        initial_configuration_rads = degs_to_rads(
            cfg['robot']['initial_configuration_degs'])
        place_configuration_rads = degs_to_rads(
            cfg['robot']['goal_configuration_degs'])

        if which == 'pick':
            self.task_generator = PickTaskGenerator(
                self.cfg,
                initial_configuration_rads,
                **cfg['bin'],
                random_seed=0,
                pool_size=n_proc)
        else:
            self.task_generator = PlaceTaskGenerator(
                self.cfg,
                initial_configuration_rads,
                place_configuration_rads,
                **cfg['bin'],
                random_seed=0,
                pool_size=n_proc)

        self.test_frequency = 1

    def get_birrt_demonstrations(self, jobs):
        jobs = [(self.cfg, discretization_step, *j) for j in jobs]

        start = time.perf_counter()
        results = Pool(n_proc).map(do_birrt_job, iter(jobs))
        end = time.perf_counter()
        self.logger.info(
            f'Got {len(results)} demonstrations in {end - start} seconds. '
            f'{(end - start) / len(results)} s/demo')

        oracle_paths = {
            'initial': torch.cat([r['initial'] for r in results]),
            'goal': torch.cat([r['goal'] for r in results]),
            'path': [r['path'][0] for r in results],
            'success': torch.cat([r['success'] for r in results]),
            'time': torch.cat([r['time'] for r in results]),
            'checks': torch.cat([r['checks'] for r in results])
        }

        if torch.any(oracle_paths['success']):
            goals = torch.cat([
                g.view(1, 6).repeat(p.shape[0] - 1, 1)
                for g, p, s
                in zip(oracle_paths['goal'],
                       oracle_paths['path'],
                       oracle_paths['success'])
                if s
            ], dim=0)
            states = torch.cat([
                p[:-1]
                for p, s
                in zip(oracle_paths['path'],
                       oracle_paths['success'])
                if s
            ], dim=0)
            actions = torch.cat([
                p[1:]
                for p, s
                in zip(oracle_paths['path'],
                       oracle_paths['success'])
                if s
            ], dim=0)
        else:
            goals = torch.zeros((0, 6), dtype=torch.float32)
            states = torch.zeros((0, 6), dtype=torch.float32)
            actions = torch.zeros((0, 6), dtype=torch.float32)

        return oracle_paths, goals, states, actions

    def get_tasks(self, n):
        start = time.perf_counter()
        tasks = self.task_generator.parallel_generate(n)
        end = time.perf_counter()

        self.logger.info(
            f'Got {len(tasks)} tasks in {end - start} seconds. '
            f'{(end - start) / len(tasks)} s/task')
        return tasks

    def get_initial_demonstrations(self, n):
        tasks = self.get_tasks(n)
        return self.get_birrt_demonstrations(tasks)

    def train_policy(
            self,
            i,
            goals,
            states,
            actions,
            goals_val,
            states_val,
            actions_val):

        self.logger.info(f'Training policy {i}')

        output_folder = self.output_folder / str(i)
        output_folder.mkdir(parents=True, exist_ok=True)

        tensorboard = torch.utils.tensorboard.SummaryWriter(str(output_folder))

        self.logger.info(f'Training on goals: {goals.shape}, states: '
                         f'{states.shape}, actions: {actions.shape}')
        steps = actions - states
        step_magnitude = torch.norm(steps, dim=1).mean()
        self.logger.info(f'Average step magnitude of training data: '
                         f'{step_magnitude}')
        torch.save(goals, output_folder / 'goals.pt')
        torch.save(states, output_folder / 'states.pt')
        torch.save(actions, output_folder / 'actions.pt')

        train_data_set = MPDataSet(goals, states, actions)
        val_data_set = MPDataSet(goals_val, states_val, actions_val)
        train_data_loader = DataLoader(
            train_data_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True)
        val_data_loader = DataLoader(
            val_data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False)

        model = get_baxter_mlp(12, 6, self.cfg['model']['dropout'])
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(training_epochs):
            model = model.train()

            metrics = defaultdict(list)
            for batch in iter(train_data_loader):
                # (b, 6), (b, 6) -> (b, 12)
                model_input = torch.cat(
                    (batch['current'], batch['goal']), dim=1)
                model_input = model_input.to(device)

                model_output = model(model_input)  # (b, 6)

                loss = F.mse_loss(
                    model_output,
                    batch['next'].to(device),
                    reduction='none')
                loss_per_batch_item = loss.mean(dim=1)  # (b, 6) -> (b,)
                mean_loss = loss_per_batch_item.mean()  # (b,) -> scalar

                optimizer.zero_grad()
                mean_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    step_magnitude = torch.norm(
                        model_output.cpu() - batch['current'], dim=1)
                    metrics['loss'].append(loss_per_batch_item.detach().cpu())
                    metrics['step_magnitude'].append(
                        step_magnitude.detach().cpu())

            mean_metrics = {
                k: torch.cat(v).mean() for k, v in metrics.items()
            }
            for k, v in mean_metrics.items():
                tensorboard.add_scalar(k, v, global_step=epoch)
            metric_strings = [f'{k}: {v}' for k, v in mean_metrics.items()]
            metric_string = ', '.join(metric_strings)
            self.logger.info(f"End epoch {epoch}, {metric_string}")
            if (epoch + 1) % save_every_n_epochs == 0:
                state_dict = model.state_dict()
                torch.save(state_dict, output_folder / f'{epoch}.pt')
                torch.save(state_dict, output_folder / 'latest.pt')

            if (epoch + 1) % validate_every_n_epochs == 0:
                # model = model.eval()

                val_losses = []
                for batch in iter(val_data_loader):
                    # (b, 6), (b, 6) -> (b, 12)
                    model_input = torch.cat(
                        (batch['current'], batch['goal']), dim=1)
                    model_input = model_input.to(device)

                    with torch.no_grad():
                        model_output = model(model_input)  # (b, 6)

                    loss = F.mse_loss(
                        model_output,
                        batch['next'].to(device),
                        reduction='none')
                    loss_per_batch_item = loss.mean(dim=1)  # (b, 6) -> (b,)
                    val_losses.append(loss_per_batch_item)
                val_losses = torch.cat(val_losses)
                mean_val_loss = val_losses.mean()
                self.logger.info(f'val_loss: {mean_val_loss}')
                tensorboard.add_scalar('val_loss', mean_val_loss,
                                       global_step=epoch)

        return model.state_dict()

    def test_policy(self, oracle_paths):
        comparison_initials = oracle_paths['initial'][:test_tasks]
        comparison_goals = oracle_paths['goal'][:test_tasks]
        comparison_tasks = zip(comparison_initials, comparison_goals)

        self.logger.info(
            f'Testing {len(comparison_initials)} tasks with NNPlanner')
        nn_paths = plan_with_generator(
            comparison_tasks,
            self.nn_planner_with_postprocessing,
            float('inf'))

        oracle_test_success = oracle_paths['success'][:test_tasks]
        nn_test_success = nn_paths['success']

        both_success = oracle_test_success & nn_test_success
        if not torch.any(both_success):
            self.logger.info('No both success!')
            return

        def get_lengths(paths, condition):
            return torch.stack([
                torch.norm(
                    paths[i][:-1] - paths[i][1:],
                    dim=1
                ).sum()
                for i in range(len(paths))
                if condition[i]
            ])

        for paths, paths_name in \
                zip([oracle_paths, nn_paths], ['ORACLE', 'NN']):
            self.logger.info('-'*40)
            self.logger.info(paths_name)
            self.logger.info(f"Success rate: {paths['success'].float().mean()}")
            self.logger.info('successful paths')
            lengths = get_lengths(paths['path'], paths['success'])
            self.logger.info(f"Length: mean {lengths.mean()} std {lengths.std()}")
            times = paths['time'][paths['success']]
            self.logger.info(f"Time: mean {times.mean()} std {times.std()}")
            checks = paths['checks'][paths['success']].float()
            self.logger.info(f"Checks: mean {checks.mean()} std {checks.std()}")
            self.logger.info('both success')
            lengths = get_lengths(paths['path'], both_success)
            self.logger.info(f"Length: mean {lengths.mean()} std {lengths.std()}")
            times = paths['time'][both_success]
            self.logger.info(f"Time: mean {times.mean()} std {times.std()}")
            checks = paths['checks'][both_success].float()
            self.logger.info(f"Checks: mean {checks.mean()} std {checks.std()}")

    def get_feedback(self, jobs):
        results = self.get_birrt_demonstrations(jobs)[0]
        print({
            k: v.shape if isinstance(v, torch.Tensor) else len(v)
            for k, v in results.items()
        })

        print([
            len(p) for p, s in zip(results['path'], results['success'])
            if s
        ])
        print([
            len(p) if p is not None else p
            for p, s in zip(results['path'], results['success'])
            if not s
        ])

        goals = results['goal'][results['success']]
        states = torch.stack([
            p[0] for p, s in zip(results['path'], results['success'])
            if s
        ], dim=0)
        actions = torch.stack([
            p[1] if len(p) > 1 else p[0]
            for p, s in zip(results['path'], results['success'])
            if s
        ], dim=0)

        return goals, states, actions

    def train(self):
        oracle_paths, goals, states, actions = \
            self.get_initial_demonstrations(initial_demonstrations)
        oracle_paths_val, goals_val, states_val, actions_val = \
            self.get_initial_demonstrations(test_tasks)
        torch.save(oracle_paths_val, self.output_folder / 'oracle_paths_val.pt')

        policies = []
        for i in range(dagger_iters):
            policy = self.train_policy(
                i, goals, states, actions, goals_val, states_val, actions_val)
            self.nn_planner_with_postprocessing.model.load_state_dict(policy)
            self.nn_planner_without_postprocessing.model.load_state_dict(policy)
            policies.append([policy])

            if (i == (dagger_iters - 1)) or (((i + 1) % self.test_frequency) == 0):
                self.test_policy(oracle_paths_val)
                if i == dagger_iters - 1:
                    # Don't rollout and collect feedback for the final
                    # iteration, as we won't be training on that data.
                    break

            self.logger.info(f'Rolling out policy {i}, {rollouts} times')
            rollout_tasks = self.get_tasks(rollouts)
            rollout = plan_with_generator(
                rollout_tasks,
                self.nn_planner_without_postprocessing,
                rollouts)
            self.logger.info('Done.')

            rollout_states = torch.cat([p[:-1] for p in rollout['path']], dim=0)
            rollout_goals = torch.cat([
                torch.tensor(g).view(1, 6).repeat(p.shape[0] - 1, 1)
                for g, p in zip(rollout['goal'], rollout['path'])
            ], dim=0)
            sampled_state_idxs = torch.randint(
                0, rollout_states.shape[0], (num_states_per_iter,))
            sampled_states = rollout_states[sampled_state_idxs]
            sampled_goals = rollout_goals[sampled_state_idxs]

            self.logger.info('Collecting interactive feedback.')
            feedback_goals, feedback_states, feedback_actions = \
                self.get_feedback(zip(sampled_states, sampled_goals))
            goals = torch.cat((goals, feedback_goals))
            states = torch.cat((states, feedback_states))
            actions = torch.cat((actions, feedback_actions))


def main():
    args = parser_args()
    which = args.which

    output_folder = Path('models') / which / args.output
    trainer = DaggerTrainer(args.config, output_folder, which)
    trainer.train()


if __name__ == '__main__':
    main()
