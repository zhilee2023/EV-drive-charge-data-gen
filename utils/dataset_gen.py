from torch.utils.data import Dataset
import torch
import json

class TimeSeriesDataset(Dataset):
    def __init__(self, df, time_steps):
        self.df = df
        self.time_steps = time_steps
        self.groups = [group[1] for group in df.groupby('seq_code') if len(group[1]) >= self.time_steps]

    def __len__(self):
        return sum([len(group) - self.time_steps + 1 for group in self.groups])

    def __getitem__(self, idx):
        for group in self.groups:
            if len(group) - self.time_steps + 1 > idx:
                x=torch.tensor(group[['trip_kind','end_index','start_hour','distance','duration',\
                                      'end_soc','stay','start_index','start_soc', 'label','battery_capacity','weekday','month']].iloc[idx:idx+self.time_steps].to_numpy(),\
                                          dtype=torch.float32)
                x[:,[2,3,4,5,6]]=torch.clamp(x[:,[2,3,4,5,6]],1e-15,1-1e-15)
                return x
            idx -= (len(group) - self.time_steps + 1)


class InitialDataset(Dataset):
    def __init__(self, df):
        with open("battery_diction.json", 'r') as file:
            diction=json.load(file)
        df=df[['start_index', 'start_soc', 'label', 'battery_capacity', 'weekday', 'month']]
        df.loc[:, 'battery_capacity'] = df['battery_capacity'].apply(lambda x: diction[str(x)])
        self.df = df
    def __len__(self):
        # Total number of possible starts for sequences of length t
        return len(self.df)

    def __getitem__(self, idx):
        # Randomly select a starting index
        # Fetch a sequence of length t starting from a random start index
        data = self.df.iloc[idx]
        # Select features and convert to tensor
        return torch.tensor(data.to_numpy(), dtype=torch.float32)


class InitialDataset2(Dataset):
    def __init__(self, df):
        df=df[['start_index', 'start_soc', 'label', 'battery_capacity', 'weekday', 'month']]
        #df.loc[:, 'battery_capacity'] = df['battery_capacity'].apply(lambda x: diction[str(x)])
        self.df = df
    def __len__(self):
        # Total number of possible starts for sequences of length t
        return len(self.df)

    def __getitem__(self, idx):
        # Randomly select a starting index
        # Fetch a sequence of length t starting from a random start index
        data = self.df.iloc[idx]
        # Select features and convert to tensor
        return torch.tensor(data.to_numpy(), dtype=torch.float32)