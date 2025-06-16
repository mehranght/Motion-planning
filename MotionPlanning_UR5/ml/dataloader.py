import torch.utils.data


class MPDataSet(torch.utils.data.Dataset):
    def __init__(self, goals, states, actions):
        self.goals = goals
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.goals)

    def __getitem__(self, item):
        return {
            'current': self.states[item],
            'next': self.actions[item],
            'goal': self.goals[item]
        }
