"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    datasets.py
"""
import torch
import numpy as np
import random
import glob
import os
from PIL import Image
from torchvision import transforms

IMAGE_DIMENSION = 100

def create_arrays_from_image(image_array: np.ndarray, offset: tuple, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    # Bild shape von (H, W, C) -> (C, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))
    
    known_array = np.zeros_like(image_array)
    # [cite_start]Gitter erstellen: offset und spacing anwenden [cite: 5]
    known_array[:, offset[1]::spacing[1], offset[0]::spacing[0]] = 1

    # [cite_start]Alles was nicht "known" ist, wird auf 0 (schwarz) gesetzt [cite: 19]
    image_array[known_array == 0] = 0

    # Maske auf 1 Channel reduzieren (spart Speicher, da RGB Maske identisch ist)
    known_array = known_array[0:1]
    
    return image_array, known_array

def resize(img: Image):
    # [cite_start]Resize + Augmentation (RandomFlip) für besseres Lernen [cite: 41]
    resize_transforms = transforms.Compose([
        transforms.Resize(size=IMAGE_DIMENSION),
        transforms.CenterCrop(size=(IMAGE_DIMENSION)),
        transforms.RandomHorizontalFlip(p=0.5) 
    ])
    return resize_transforms(img)

def preprocess(input_array: np.ndarray):
    # Normalisieren auf 0.0 - 1.0
    return np.array(input_array / 255.0, dtype=np.float32)

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder
    """

    def __init__(self, datafolder: str):
        # Alle jpg Bilder rekursiv finden
        self.imagefiles = sorted(glob.glob(os.path.join(datafolder, "**", "*.jpg"), recursive=True))

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, idx: int):
        index = int(idx)

        # Bild laden
        try:
            image = Image.open(self.imagefiles[index]).convert("RGB")
        except Exception as e:
            print(f"Fehler beim Laden von {self.imagefiles[index]}: {e}")
            # Fallback: Nimm das nächste Bild
            return self.__getitem__((index + 1) % len(self))

        image = np.array(resize(image))
        image = preprocess(image)

        # [cite_start]Zufälliges Gitter gemäß Task Description [cite: 22]
        spacing_x = random.randint(2, 6)
        spacing_y = random.randint(2, 6)
        offset_x = random.randint(0, 8)
        offset_y = random.randint(0, 8)

        spacing = (spacing_x, spacing_y)
        offset = (offset_x, offset_y)

        # Input (Löcher) und Maske erstellen
        input_array, known_array = create_arrays_from_image(image.copy(), offset, spacing)

        input_array = torch.from_numpy(input_array)
        known_array = torch.from_numpy(known_array)

        # Zusammenfügen: 3 Channel Bild + 1 Channel Maske = 4 Channel Input
        input_tensor = torch.cat((input_array, known_array), dim=0)

        # Ziel ist das originale Bild (C, H, W)
        target_image = torch.from_numpy(np.transpose(image, (2, 0, 1)))

        return input_tensor, target_image