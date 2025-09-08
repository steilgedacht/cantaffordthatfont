import glob
import numpy as np
import os
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch.utils.data import Dataset
import random
import io
from scipy.ndimage import gaussian_filter

buffer = io.BytesIO()

def load_fonts(only_basename=False):
    dir_path = os.getcwd()
    font_files = glob.glob(os.path.join(dir_path, 'fonts_subset/*.ttf'), recursive=True)
    font_files.sort()
    if only_basename:
        for i in range(len(font_files)):
            font_files[i] = os.path.splitext(os.path.basename(font_files[i]))[0]
    return font_files

def add_jpeg_artifacts(img: Image.Image, quality: int = 25) -> Image.Image:
    buffer.seek(0)
    buffer.truncate(0)
    img.save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    return Image.open(buffer).copy()

def random_cosine_texture(sigma=20.0):
    H, W = 150, 700
    
    # Step 1: White noise
    noise = np.random.rand(H, W).astype(np.float32)
    
    # Step 2: Smooth it with Gaussian blur
    texture = gaussian_filter(noise, sigma=sigma)
    
    # Step 3: Normalize to [0,1]
    texture -= texture.min()
    texture /= texture.max()
    
    return texture

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
            if basename.startswith("NotoSans"):
                basename = "NotoSans"
            elif basename.startswith("NotoSerif"):
                basename = "NotoSerif"
            self.font_files_dict[file] = basename
        
        self.font_files = font_files
        self.fonts_unique = sorted(list(set(self.font_files_dict.values())))

        self.font_to_subfonts = {}
        for font in self.font_files_dict.items():
            if font[1] not in self.font_to_subfonts:
                self.font_to_subfonts[font[1]] = []
            self.font_to_subfonts[font[1]].append(font[0])

        self.deterministic = False

    def __len__(self):
        return len(self.fonts_unique) * 100

    def __getitem__(self, idx):
        if self.deterministic:
            font_family_name = self.fonts_unique[idx]
        else:
            font_family_name = random.choice(self.fonts_unique)
    
        if len(self.font_to_subfonts[font_family_name]) != 1:
            font_family = random.choice(self.font_to_subfonts[font_family_name])
        else:
            font_family = self.font_to_subfonts[font_family_name][0]
        
        image_data = self.generate_samples(font_family)
        image_data = torch.tensor(image_data, dtype=torch.float32)

        onehot_label = torch.tensor(self.fonts_unique.index(font_family_name), dtype=torch.long)

        return image_data, onehot_label
    

    def generate_samples(self, font_family):
        text_length = random.randint(7, 18)
        text = "".join(random.choice(self.config.characters) for _ in range(text_length))

        # the images should have a size of 700px x 150px
        image_size = (700, 150)
        font_size = np.random.randint(40, 150)

        try:
            font = ImageFont.truetype(font_family, font_size)
        except:
            print(font_family)
            return

        noise = np.random.randint(200, 255, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
        noise = Image.fromarray(noise, "RGB")
        blurnoise = noise.filter(ImageFilter.GaussianBlur(radius=random.randint(1,5)))
        draw = ImageDraw.Draw(blurnoise)

        x = random.randint(0, 100)
        y = random.randint(0, 50)

        try:
            draw.text((x, y), text, fill="black", font=font)
        except:
            print(font_family)

        
        image = blurnoise.filter(ImageFilter.GaussianBlur(radius=random.randint(0,3)))

        image = add_jpeg_artifacts(image, quality=random.randint(10, 90))

        image = image.convert("L")
        image = np.asarray(image)

        image = image - image * random_cosine_texture() * random.uniform(0.0, 0.6)

        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        return image
