"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    train.py
"""
import datasets
from architecture import MyModel
from utils import plot, evaluate_model

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

def train(seed, testset_ratio, validset_ratio, data_path, results_path, early_stopping_patience, device, learningrate,
          weight_decay, n_updates, use_wandb, print_train_stats_at, print_stats_at, plot_at, validate_at, batchsize,
          network_config: dict):
    
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    if use_wandb:
        wandb.login()
        wandb.init(project="image_inpainting", config={
            "lr": learningrate, "updates": n_updates, "batch": batchsize, "arch": "U-Net L1"
        })

    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)

    # Dataset laden
    image_dataset = datasets.ImageDataset(datafolder=data_path)
    n_total = len(image_dataset)
    n_test = int(n_total * testset_ratio)
    n_valid = int(n_total * validset_ratio)
    n_train = n_total - n_test - n_valid
    
    indices = np.random.permutation(n_total)
    dataset_train = Subset(image_dataset, indices=indices[0:n_train])
    dataset_valid = Subset(image_dataset, indices=indices[n_train:n_train + n_valid])
    dataset_test = Subset(image_dataset, indices=indices[n_train + n_valid:n_total])

    dataloader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    network = MyModel(**network_config).to(device)
    network.train()

    # --- WICHTIGE ÄNDERUNG: L1 LOSS ---
    # L1 Loss sorgt für schärfere Bilder als MSE. 
    # MSE akzeptiert Unschärfe, L1 zwingt zu harten Entscheidungen.
    criterion = torch.nn.L1Loss() 
    
    # Für die Berechnung des Scores (RMSE) brauchen wir trotzdem MSE
    mse_metric = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=learningrate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    i = 0
    counter = 0
    best_loss = np.inf
    saved_model_path = os.path.join(results_path, "best_model.pt")

    print(f"Training start on {device} with L1 Loss")

    while i < n_updates:
        for inputs, targets in dataloader_train:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            
            # Training mit L1 Loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (i + 1) % print_train_stats_at == 0:
                print(f'Step {i + 1}: L1 Loss: {loss.item():.5f}')

            if use_wandb and (i+1) % print_stats_at == 0:
                wandb.log({"train/loss": loss.item()}, step=i)

            if (i + 1) % plot_at == 0:
                plot(inputs.cpu().numpy(), targets.cpu().numpy(), outputs.detach().cpu().numpy(), plotpath, i)

            # --- VALIDATION ---
            if (i + 1) % validate_at == 0:
                # Evaluate nutzt MSE für die RMSE Berechnung (wie in der Challenge)
                val_mse, val_rmse = evaluate_model(network, dataloader_valid, mse_metric, device)
                print(f"Validation @ {i+1}: RMSE={val_rmse:.4f}")
                
                scheduler.step(val_mse) # Scheduler hört auf MSE (das echte Ziel)

                if val_mse < best_loss:
                    best_loss = val_mse
                    torch.save(network.state_dict(), saved_model_path)
                    print(f"--> New Best Model! RMSE: {val_rmse:.4f}")
                    counter = 0
                else:
                    counter += 1
                    print(f"No improvement. ({counter}/{early_stopping_patience})")

                if counter >= early_stopping_patience:
                    print("Early Stopping.")
                    i = n_updates
                    break

            i += 1
            if i >= n_updates: break

    # Final Test
    print("Testing Best Model...")
    network.load_state_dict(torch.load(saved_model_path))
    test_loss, test_rmse = evaluate_model(network, dataloader_test, mse_metric, device)
    print(f'Final Testset RMSE: {test_rmse}')