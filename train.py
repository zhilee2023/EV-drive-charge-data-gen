from utils.model_build import get_dataloader, build_model, load_config
from utils.train_utils import train_epoch, evaluate_and_save
import torch
import time
import os
import numpy as np
import datetime
import argparse

def train(config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(config_path)
    t = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    output_dir = f"model_{t}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/output_{t}.txt"

    dataloader, dataloader_initial = get_dataloader(config)
    model, optimizers, schedulers, discriminator_model, optimizers_D, schedulers_D = build_model(config,device)

    all_losses = []
    start_time = time.time()

    for epoch in range(config['num_epochs']):
        epoch_loss = train_epoch(
            model, dataloader, dataloader_initial, optimizers, epoch, schedulers,
            output_file, config, discriminator_model, optimizers_D, schedulers_D
        )
        all_losses.append(epoch_loss)

    torch.save(model.state_dict(), f"{output_dir}/model_final.pth")
    np.save(f"{output_dir}/loss.npy", all_losses)

    elapsed_time = (time.time() - start_time) / 60
    with open(output_file, 'a') as f:
        f.write(f"Training completed in {elapsed_time:.2f} minutes\n")
        f.write("Config used:\n")
        f.write(str(config)+"\n")
    evaluate_and_save(model, output_dir, output_file, config, t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()
    train(args.config)
