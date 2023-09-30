import glob
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

def load_fonts(only_basename=False):
    dir_path = os.getcwd()
    font_files = glob.glob(os.path.join(dir_path, 'fonts_subset/*.ttf'), recursive=True)
    font_files.sort()
    if only_basename:
        for i in range(len(font_files)):
            font_files[i] = os.path.splitext(os.path.basename(font_files[i]))[0]
    return font_files

class Datasubsets(Dataset):
    def __init__(self):
        super().__init__()

        dir_path = os.getcwd()
        files = glob.glob(os.path.join(dir_path, 'data/**/*.jpg'), recursive=True)
        if len(files) == 0:
            raise ValueError

        # transform all paths to absolute paths
        self.dataset = []
        for f in files:
            self.dataset.append(os.path.abspath(f))
        self.dataset.sort()
        self.embedding = load_fonts(only_basename=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # load the images from the path
        image_path = self.dataset[idx]
        font_name = os.path.basename(os.path.dirname(image_path))

        image_data = Image.open(image_path)
        image_data = image_data.convert("L")

        # Convert to float32
        image_data = np.asarray(image_data, dtype=np.float32)
        image_data = torch.tensor(image_data)

        onehot_label = [0.01] * len(self.embedding)
        onehot_label[self.embedding.index(font_name)] = 0.99
        onehot_label = torch.tensor(np.asarray(onehot_label, dtype=np.float32))

        return image_data, onehot_label