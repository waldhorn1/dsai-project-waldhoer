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
          continue_training, network_config: dict): # <--- NEU: continue_training Parameter
    
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    # Pfade
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)
    saved_model_path = os.path.join(results_path, "best_model.pt")
    checkpoint_path = os.path.join(results_path, "checkpoint.pt") # <--- NEU: Pfad für Checkpoint

    # WandB
    if use_wandb:
        wandb.login()
        wandb.init(project="image_inpainting", config={
            "lr": learningrate, "updates": n_updates, "batch": batchsize, "arch": "U-Net L1"
        }, resume=continue_training) # <--- NEU: Resume Info an WandB

    # Dataset Setup (wie gehabt)
    image_dataset = datasets.ImageDataset(datafolder=data_path)
    n_total = len(image_dataset)
    n_test = int(n_total * testset_ratio)
    n_valid = int(n_total * validset_ratio)
    n_train = n_total - n_test - n_valid
    
    indices = np.random.permutation(n_total)
    dataset_train = Subset(image_dataset, indices=indices[0:n_train])
    dataset_valid = Subset(image_dataset, indices=indices[n_train:n_train + n_valid])
    dataset_test = Subset(image_dataset, indices=indices[n_train + n_valid:n_total])

    dataloader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Model, Loss, Optimizer
    network = MyModel(**network_config).to(device)
    network.train()

    criterion = torch.nn.L1Loss() 
    mse_metric = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=learningrate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Variablen initialisieren
    i = 0
    counter = 0
    best_loss = np.inf

    # --- RESUME LOGIC ---
    if continue_training and os.path.exists(checkpoint_path):
        print(f"Lade Checkpoint von {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        i = checkpoint['step']
        best_loss = checkpoint['best_loss']
        counter = checkpoint['early_stopping_counter']
        
        print(f"Training wird fortgesetzt bei Schritt {i}. Best Loss bisher: {best_loss:.4f}")
    else:
        print(f"Starte neues Training auf {device}")

    # Training Loop
    while i < n_updates:
        for inputs, targets in dataloader_train:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (i + 1) % print_train_stats_at == 0:
                print(f'Step {i + 1}/{n_updates}: L1 Loss: {loss.item():.5f}')

            if use_wandb and (i+1) % print_stats_at == 0:
                wandb.log({"train/loss": loss.item()}, step=i)

            if (i + 1) % plot_at == 0:
                plot(inputs.cpu().numpy(), targets.cpu().numpy(), outputs.detach().cpu().numpy(), plotpath, i)

            # --- VALIDATION & SAVING ---
            if (i + 1) % validate_at == 0:
                val_mse, val_rmse = evaluate_model(network, dataloader_valid, mse_metric, device)
                print(f"Validation @ {i+1}: RMSE={val_rmse:.4f}")
                
                scheduler.step(val_mse)

                # Best Model speichern (für Abgabe)
                if val_mse < best_loss:
                    best_loss = val_mse
                    torch.save(network.state_dict(), saved_model_path)
                    print(f"--> New Best Model! RMSE: {val_rmse:.4f}")
                    counter = 0
                else:
                    counter += 1
                    print(f"No improvement. ({counter}/{early_stopping_patience})")

                # --- CHECKPOINT SPEICHERN (Jedes Mal bei Validierung) ---
                checkpoint = {
                    'step': i + 1,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'early_stopping_counter': counter
                }
                torch.save(checkpoint, checkpoint_path)
                # print("Checkpoint saved.") 

                if counter >= early_stopping_patience:
                    print("Early Stopping.")
                    i = n_updates
                    break

            i += 1
            if i >= n_updates: break

    # Final Test
    print("Testing Best Model...")
    if os.path.exists(saved_model_path):
        network.load_state_dict(torch.load(saved_model_path))
        test_loss, test_rmse = evaluate_model(network, dataloader_test, mse_metric, device)
        print(f'Final Testset RMSE: {test_rmse}')