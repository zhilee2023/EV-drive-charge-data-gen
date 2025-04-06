from utils.model_build import build_model, load_config
import torch
import pandas as pd
import numpy as np
import argparse
import os
import datetime

def sample(config_path, model_path):
    #time_steps=60
    batch_size=256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(config_path)
    time_steps=config['time_steps']
    model, *_ = build_model(config,device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data = np.zeros((0, config['x_dim']))
    for _ in range(config['sample_batch_num']):
        tt,log_prob,_=model.sample(batch_size,time_steps,initial=0,method=None,c0=None)
        new_array =tt.reshape(tt.shape[0]*tt.shape[1],-1)
        data=np.concatenate([data,new_array],axis=0)
        # if step>=sample_num:
        #     break;
    t = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    output_dir = f"model_{t}_samples"
    os.makedirs(output_dir, exist_ok=True)
    #output_file = f"{output_dir}/output_{t}.txt"
    column = ["trip_kind", "end_index", "start_hour", "distance", "duration", "end_soc", "stay",
              "start_index", "start_soc", "label", "battery_capacity", "weekday", "month"]
    samples_df = pd.DataFrame(data).dropna()
    samples_df.columns = column
    samples_df.to_csv(f"{output_dir}/samples.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    #parser.add_argument('--output_dir', type=str, required=True, help='Directory to save samples')
    args = parser.parse_args()
    sample(args.config, args.model_path)
