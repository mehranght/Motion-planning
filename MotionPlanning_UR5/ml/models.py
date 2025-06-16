from torch import nn


def get_baxter_mlp(input_size, output_size, dropout_rates):
    """ The official MLP from https://github.com/anthonysimeonov/baxter_mpnet_experiments/blob/55a1cac928874bf41db04f71bc8419f021c55b09/architectures.py#L9

    :param input_size: probably 12
    :param output_size: probably 6
    :param dropout_rate: p for nn.Dropout
    :return: model
    """
    return nn.Sequential(
        nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(p=dropout_rates[0]),
        nn.Linear(1280, 896), nn.PReLU(), nn.Dropout(p=dropout_rates[1]),
        nn.Linear(896, 512), nn.PReLU(), nn.Dropout(p=dropout_rates[2]),
        nn.Linear(512, 384), nn.PReLU(), nn.Dropout(p=dropout_rates[3]),
        nn.Linear(384, 256), nn.PReLU(), nn.Dropout(p=dropout_rates[4]),
        nn.Linear(256, 128), nn.PReLU(), nn.Dropout(p=dropout_rates[5]),
        nn.Linear(128, 64), nn.PReLU(), nn.Dropout(p=dropout_rates[6]),
        nn.Linear(64, 32), nn.PReLU(), nn.Dropout(p=dropout_rates[7]),
        nn.Linear(32, output_size)
    )
