from utils.model_build import get_dataloader, build_model, load_config
from utils.train_utils import fine_tune_cpo
import torch
import argparse
import os
import datetime

def fine_tune(config_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(config_path)
    dataloader, _ = get_dataloader(config)
    model, *_ = build_model(config,device)
    model.train()
    model.load_state_dict(torch.load(model_path))
    t = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    output_dir = f"model_{t}_samples"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/output_finetune.txt"
    fine_tune_cpo(model, dataloader, output_dir, output_file, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    #parser.add_argument('--output_dir', type=str, required=True, help='Directory to save finetuned model')
    args = parser.parse_args()
    fine_tune(args.config, args.model_path)
