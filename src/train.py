"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    utils.py
"""

import torch
import numpy as np
import os

from architecture import MyModel


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    from matplotlib import pyplot as plt
    
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            img = data[i:i + 1:, 0:3, :, :]
            img = np.squeeze(img)
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update + 1:07d}_{i + 1:02d}.jpg"))

    plt.close(fig)


def testset_plot(input_array, output_array, path, index):
    """Plotting the inputs, targets and predictions to file `path` for testset"""
    from matplotlib import pyplot as plt
    
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    for ax, data, title in zip(axes, [input_array, output_array], ["Input", "Prediction"]):
        ax.clear()
        ax.set_title(title)
        img = data[0:3, :, :] # Nur die ersten 3 Channel (RGB) anzeigen
        img = np.squeeze(img)
        if len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_axis_off()
    fig.savefig(os.path.join(path, f"testset_{index + 1:07d}.jpg"))

    plt.close(fig)


def evaluate_model(network: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Returnse MSE and RMSE of the model on the provided dataloader"""
    network.eval()
    loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            input_array, target = data
            input_array = input_array.to(device)
            target = target.to(device)

            outputs = network(input_array)

            loss += loss_fn(outputs, target).item()

        loss = loss / len(dataloader)

        network.train()

        return loss, 255.0 * np.sqrt(loss)


def read_compressed_file(file_path: str):
    with np.load(file_path) as data:
        input_arrays = data['input_arrays']
        known_arrays = data['known_arrays']
    return input_arrays, known_arrays


def create_predictions(model_config, state_dict_path, testset_path, device, save_path, plot_path, plot_at=20):
    """
    Erstellt Predictions und überschreibt bekannte Pixel mit den Originalwerten (Masking-Trick).
    """

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    model = MyModel(**model_config)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()

    input_arrays, known_arrays = read_compressed_file(testset_path)

    known_arrays = known_arrays.astype(np.float32)
    input_arrays_norm = input_arrays.astype(np.float32) / 255.0

    # Input für Netzwerk: Concatenate Image + Mask
    network_input = np.concatenate((input_arrays_norm, known_arrays), axis=1)

    predictions = list()

    with torch.no_grad():
        for i in range(len(network_input)):
            print(f"Processing image {i + 1}/{len(network_input)}")
            
            # 1. Prediction vom Model holen
            inp_tensor = torch.from_numpy(network_input[i:i+1]).to(device)
            output = model(inp_tensor)
            output = output.cpu().numpy() # Shape (1, 3, 100, 100)
            
            # Prediction aus Batch holen
            pred_img = output[0]
            
            # 2. Originaldaten holen
            orig_img = input_arrays_norm[i] # Das Bild mit schwarzen Löchern (0-1)
            mask = known_arrays[i]          # Maske (1=bekannt, 0=unbekannt)
            
            # 3. Post-Processing: Überschreiben der bekannten Pixel
            # Wir nehmen die Prediction nur dort, wo die Maske 0 ist.
            # Ansonsten nehmen wir das Originalbild.
            final_img = (pred_img * (1 - mask)) + orig_img
            
            predictions.append(final_img)

            if (i + 1) % plot_at == 0:
                testset_plot(orig_img, final_img, plot_path, i)

    predictions = np.stack(predictions, axis=0)

    predictions = (np.clip(predictions, 0, 1) * 255.0).astype(np.uint8)

    data = {
        "predictions": predictions
    }

    np.savez_compressed(save_path, **data)

    print(f"Predictions saved at {save_path}")