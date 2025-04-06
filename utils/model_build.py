import torch
import pandas as pd
import json
import os
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler

from .TransformerBayes import TransformerBayes
from .TransformerVAE import TransformerVAE
from .TransformerWGAN import TransformerWGAN
from .TransformerGibbs import TransformerGibbs
from .discriminator import discriminator
from .util import data_preprocess_test
from .dataset_gen import TimeSeriesDataset, InitialDataset


def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_dataloader(config):
    df = pd.read_csv("samples.csv")
    if config['filter_veh']:
        filter_df = pd.read_json('ratio_within_center.json')
        filtered_codes = filter_df[filter_df['within_center'] > 0.75].index
        df = df[df['seq_code'].isin(filtered_codes)]

    df = data_preprocess_test(df)

    dataset = TimeSeriesDataset(df, time_steps=config['time_steps'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    dataloader_initial = DataLoader(InitialDataset(df), batch_size=config['batch_size'], shuffle=True, drop_last=True)
    return dataloader, dataloader_initial

def build_model(config,device):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_classes = {
        'TransformerBayes': TransformerBayes,
        'TransformerVAE': TransformerVAE,
        'TransformerWGAN': TransformerWGAN,
        'TransformerGibbs': TransformerGibbs
    }

    model_class = model_classes[config['model_type']]
    model = model_class(
        x_dim=config['x_dim'],  time_step=config['time_steps'], n_head=config['n_head'], n_layers=config['n_layers'],\
        d_model=config['d_model'],n_loc=config['n_loc'], embed_vector_len=config['embed_vector_len'], device=device, tau=config['tau'], rnn_backbone=config['rnn_backbone']
    )

    optimizers = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config['learning_rate'], weight_decay=config['decay_rate'])
    schedulers = lr_scheduler.CosineAnnealingLR(optimizers, T_max=config['num_epochs'])

    discriminator_model, optimizers_D, schedulers_D = None, None, None
    if config['model_type'] == "TransformerWGAN":
        discriminator_model = discriminator(config['x_dim'], config['time_steps'],
                                            d_model=config['d_model'],
                                            embedding_len=config['embed_vector_len'],
                                            device=device)
        optimizers_D = optim.Adam(filter(lambda p: p.requires_grad, discriminator_model.parameters()),
                                  lr=config['learning_rate'])
        schedulers_D = lr_scheduler.LinearLR(optimizers_D, start_factor=1.0, end_factor=0.5, total_iters=10)

    return model, optimizers, schedulers, discriminator_model, optimizers_D, schedulers_D
