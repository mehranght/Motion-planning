import argparse
from collections import defaultdict
from pathlib import Path


import torch
import torch.nn.functional as F
import torch.utils.tensorboard
from torch.utils.data.dataloader import DataLoader
from ml.dataloader import MPDataSet
from ml.models import get_baxter_mlp


device = 'cuda:0'
learning_rate = 0.0001
save_every_n_epochs = 1
validate_every_n_epochs = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data', default='train_data_place.pt')
    parser.add_argument('--val-data', default='val_data_place.pt')
    parser.add_argument('--output-dir', default='./models')
    parser.add_argument('--n-epoch', type=int, default=8000)
    return parser.parse_args()


def main(batch_size, output_dir, goals, states, actions, goals_val, states_val, actions_val, n_epoch):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    tensorboard = torch.utils.tensorboard.SummaryWriter(str(output_dir))

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

    model = get_baxter_mlp(12, 6, dropout_rate=0.0)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoch):
        model = model.train()

        metrics = defaultdict(list)
        for batch in iter(train_data_loader):
            # (b, 6), (b, 6) -> (b, 12)
            model_input = torch.cat((batch['current'], batch['goal']), dim=1)
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
                metrics['step_magnitude'].append(step_magnitude.detach().cpu())

        mean_metrics = {
            k: torch.cat(v).mean() for k, v in metrics.items()
        }
        for k, v in mean_metrics.items():
            tensorboard.add_scalar(k, v, global_step=epoch)
        metric_strings = [f'{k}: {v}' for k, v in mean_metrics.items()]
        metric_string = ', '.join(metric_strings)
        print(f"End epoch {epoch}, {metric_string}")
        if (epoch + 1) % save_every_n_epochs == 0:
            state_dict = model.state_dict()
            torch.save(state_dict, output_dir / f'{epoch}.pt')
            torch.save(state_dict, output_dir / 'latest.pt')

        if (epoch + 1) % validate_every_n_epochs == 0:
            # model = model.eval()

            val_losses = []
            for batch in iter(val_data_loader):
                # (b, 6), (b, 6) -> (b, 12)
                model_input = torch.cat((batch['current'], batch['goal']),
                                        dim=1)
                model_input = model_input.to(device)

                with torch.no_grad():
                    model_output = model(model_input)  # (b, 6)

                loss = F.mse_loss(model_output, batch['next'].to(device),
                                  reduction='none')
                loss_per_batch_item = loss.mean(dim=1)  # (b, 6) -> (b,)
                val_losses.append(loss_per_batch_item)
            val_losses = torch.cat(val_losses)
            mean_val_loss = val_losses.mean()
            print(f'val_loss: {mean_val_loss}')
            tensorboard.add_scalar('val_loss', mean_val_loss, global_step=epoch)

    return model.state_dict()


if __name__ == '__main__':
    parsed_args = parse_args()
    main(**vars(parsed_args))