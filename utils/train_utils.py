import torch
import numpy as np
import pandas as pd
from .compare_gibbs import model_compare
from .rho_cal import calculate_rho

def train_epoch(model, dataloader, dataloader_initial, optimizers, epoch, schedulers,
                output_file, config, discriminator_model=None, optimizers_D=None, schedulers_D=None):
    model.train()
    batch_num = 0
    epoch_losses = {'recon': 0, 'kl': 0, 'cond': 0, 'dis': 0, 'gen': 0, 'p0': 0}

    for batch, init_batch in zip(dataloader, dataloader_initial):
        batch, init_batch = batch.to(model.device), init_batch.to(model.device)
        optimizers.zero_grad()

        if config['model_type'] == 'TransformerWGAN':
            optimizers_D.zero_grad()
            p0_loss, recon_loss, cond_loss, (real_sample, real_mask, fake_sample, fake_mask) = model(batch, init_batch)
            real_score = discriminator_model(real_sample.detach(), real_mask)
            fake_score = discriminator_model(fake_sample.detach(), fake_mask)
            dis_loss = -torch.mean(real_score - fake_score)
            dis_loss.backward()
            optimizers_D.step()

            for p in discriminator_model.parameters():
                p.data.clamp_(-0.01, 0.01)

            gen_loss = -torch.mean(discriminator_model(fake_sample, fake_mask))
            loss = p0_loss + recon_loss + cond_loss + gen_loss
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
            optimizers.step()

            epoch_losses['dis'] += dis_loss.item()
            epoch_losses['gen'] += gen_loss.item()

        elif config['model_type'] == 'TransformerVAE':
            p0_loss, recon_loss, kl_loss, cond_loss = model(batch, init_batch)
            loss = recon_loss + kl_loss + cond_loss + p0_loss
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)
            optimizers.step()
            epoch_losses['kl'] += kl_loss.item()

        else:
            p0_loss, recon_loss, cond_loss = model(batch, init_batch)
            loss = p0_loss + recon_loss + cond_loss
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)
            optimizers.step()

        epoch_losses['recon'] += recon_loss.item()
        epoch_losses['cond'] += cond_loss.item()
        epoch_losses['p0'] += p0_loss.item()
        batch_num += 1

        if batch_num >= config['batch_num_per_epoch']:
            break

    schedulers.step()
    if config['model_type'] == 'TransformerWGAN':
        schedulers_D.step()

    for key in epoch_losses:
        epoch_losses[key] /= (batch_num + 1)

    with open(output_file, 'a') as f:
        f.write(f"Epoch_{epoch}: {epoch_losses}\n")

    return epoch_losses

def fine_tune_cpo(model, dataloader, output_dir, output_file, config):
    # cloned_model = copy.deepcopy(model)
    # for para in cloned_model.parameters():
    #     para.requires_grad = False

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-8)

    for epoch in range(config['cpo_epoch']):
        epoch_loss = 0
        batch_num = 0

        for batch in dataloader:
            samples = model.sample_one_step(batch.to(model.device), 40)
            batch_num += 1

            for ss in samples:
                i, j = ss
                a1 = model.log_prob_forward(batch.to(model.device), i).mean(axis=-1)
                a2 = model.log_prob_forward(batch.to(model.device), j).mean(axis=-1)
                loss = -(torch.log(torch.sigmoid(config['beta'] * a1 - config['beta'] * a2))).mean() / len(samples) - config['lambda_'] * a1.mean() / len(samples)

                optimizer_ft.zero_grad()
                loss.backward()
                optimizer_ft.step()
                epoch_loss += loss.item()

            if batch_num > config['cpo_batch_num']:
                break

        epoch_loss /= config['cpo_batch_num']
        with open(output_file, 'a') as f:
            f.write(f"Fine-tune Epoch_{epoch} Loss: {epoch_loss}\n")

    model_path = f"{output_dir}/model_finetune.pth"
    torch.save(model.state_dict(), model_path)
    rho1, rho2 = model_compare(config['sample_batch_num'], config['sample_batch_size'], model, output_dir,
                               time_steps=config['time_steps'], initial=0, save="samples_fine_tune.csv", filter=False)
    with open(output_file, 'a') as f:
        f.write(f"Fine-tune Evaluation rho1={rho1}, rho2={rho2}\n")

def evaluate_and_save(model, output_dir, output_file, config, t):
    if config['model_type'] == 'TransformerGibbs':
        rho1, rho2 = model_compare(config['sample_batch_num'], config['sample_batch_size'], model, output_dir,
                                   time_steps=config['time_steps'], initial=0, save="samples.csv")
    else:
        data = np.zeros((0, config['x_dim']))
        for _ in range(config['sample_batch_num']):
            samples = model.sample(256, config['time_steps'])
            new_array = samples.reshape(samples.shape[0] * samples.shape[1], -1)
            data = np.concatenate([data, new_array], axis=0)
        column = ["trip_kind", "end_index", "start_hour", "distance", "duration", "end_soc", "stay",
                  "start_index", "start_soc", "label", "battery_capacity", "weekday", "month"]
        samples_df = pd.DataFrame(data).dropna()
        samples_df.columns = column
        samples_df.to_csv(f"{output_dir}/samples.csv", index=False)
        rho1, rho2 = calculate_rho(f"{output_dir}/samples.csv")

    with open(output_file, 'a') as f:
        f.write(f"Evaluation rho1={rho1}, rho2={rho2}\n")
