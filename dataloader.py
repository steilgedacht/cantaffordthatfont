import glob
import numpy as np
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
import random


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
        image_data = np.asarray(image_data)
        image_data = torch.tensor(image_data, dtype=torch.float32)

        onehot_label = torch.tensor(self.embedding.index(font_name), dtype=torch.long)

        return image_data, onehot_label
    

class Datagenerator(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config

        dir_path = os.getcwd()
        font_files = glob.glob(os.path.join(dir_path, f'{config.fonts_folder}/*.ttf'), recursive=True)
        font_files.sort()
        self.font_files_dict = {}
        for file in font_files:
            basename = os.path.splitext(os.path.basename(file))[0]
            basename = basename.split("[")[0].split("-")[0]
            self.font_files_dict[file] = basename
        
        self.font_files = font_files
        self.fonts_unique = sorted(list(set(self.font_files_dict.values())))

        self.font_to_subfonts = {}
        for font in self.font_files_dict.items():
            if font[1] not in self.font_to_subfonts:
                self.font_to_subfonts[font[1]] = []
            self.font_to_subfonts[font[1]].append(font[0])

    def __len__(self):
        return len(self.fonts_unique) * 100

    def __getitem__(self, idx):
        font_family_name = random.choice(self.fonts_unique)
        if len(self.font_to_subfonts[font_family_name]) != 1:
            font_family = random.choice(self.font_to_subfonts[font_family_name])
        else:
            font_family = self.font_to_subfonts[font_family_name][0]
        
        image_data = self.generate_samples(font_family)
        image_data = image_data.convert("L")
        image_data = np.asarray(image_data)
        image_data = torch.tensor(image_data, dtype=torch.float32)

        onehot_label = torch.tensor(self.fonts_unique.index(font_family_name), dtype=torch.long)

        return image_data, onehot_label
    

    def generate_samples(self, font_family):
        text_length = random.randint(7, 18)
        text = "".join(random.choice(self.config.characters) for _ in range(text_length))

        # the images should have a size of 700px x 150px
        image_size = (700, 150)
        font_size = 70

        try:
            font = ImageFont.truetype(font_family, font_size)
        except:
            print(font_family)

        # noise = np.random.randint(0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
        noise = 255 * np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) 
        image = Image.fromarray(noise, "RGB")
        draw = ImageDraw.Draw(image)

        x = random.randint(0, abs(image_size[0] - font_size * text_length))
        y = random.randint(0, image_size[1] - font_size)

        try:
            draw.text((x, y), text, fill="black", font=font)
        except:
            print(font_family)

    
        return image
