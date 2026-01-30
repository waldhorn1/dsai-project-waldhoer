"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    main.py
"""

import os
from utils import create_predictions
from train import train


if __name__ == '__main__':
    config_dict = dict()

    config_dict['seed'] = 42
    config_dict['testset_ratio'] = 0.1
    config_dict['validset_ratio'] = 0.1
    config_dict['results_path'] = os.path.join("results")
    config_dict['data_path'] = os.path.join("data", "dataset")
    config_dict['device'] = "cuda" 

    config_dict['learningrate'] = 1e-3
    config_dict['weight_decay'] = 1e-5
    config_dict['n_updates'] = 100000
    config_dict['batchsize'] = 32
    config_dict['early_stopping_patience'] = 25 
    config_dict['use_wandb'] = False

    # --- NEU: Fortsetzen aktivieren ---
    # Setze auf True, um beim letzten Checkpoint weiterzumachen.
    # Setze auf False, um komplett von vorne zu beginnen (überschreibt alten Checkpoint).
    config_dict['continue_training'] = True 

    config_dict['print_train_stats_at'] = 50
    config_dict['print_stats_at'] = 100
    config_dict['plot_at'] = 500
    config_dict['validate_at'] = 500

    network_config = {
        'n_in_channels': 4
    }
    
    config_dict['network_config'] = network_config

    train(**config_dict)
    
    # ... (Rest bleibt gleich für Predictions)
    testset_path = os.path.join("data", "challenge_testset.npz")
    state_dict_path = os.path.join(config_dict['results_path'], "best_model.pt")
    submission_folder = os.path.join(config_dict['results_path'], "testset")
    os.makedirs(submission_folder, exist_ok=True)
    save_path = os.path.join(submission_folder, "my_submission_unet.npz")
    plot_path = os.path.join(submission_folder, "plots")

    print("Erstelle Predictions für das Testset...")
    create_predictions(config_dict['network_config'], 
                       state_dict_path, 
                       testset_path, 
                       config_dict['device'], 
                       save_path, 
                       plot_path, 
                       plot_at=50)