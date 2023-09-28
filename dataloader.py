import glob
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from train import load_fonts

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
        image_data = self.dataset[idx]
        # print(idx)
        image_data = Image.open(image_data)

        # Convert to float32
        image_data = np.asarray(image_data, dtype=np.float32)
        image_data = torch.tensor(image_data)

        onehot_label = [0.01] * len(self.embedding)
        onehot_label[idx] = 0.99

        return image_data, onehot_label, idx