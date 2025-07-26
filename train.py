import numpy as np
import os
import glob
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import wandb

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataloader import Datasubsets, Datagenerator
from torchvision import models
from tqdm import tqdm


class Config:
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    output_classes = 1738 
    num_epochs = 100
    batch_size = 100
    learning_rate = 7e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_data = False  # Set to True to generate data
    dropout = 0.2
    wandb_tag = "resnet18"
    fonts_folder = "all_fonts_filtered" 
config = Config()


def load_fonts(only_basename=False):
    dir_path = os.getcwd()
    font_files = glob.glob(os.path.join(dir_path, f'{config.fonts_folder}/*.ttf'), recursive=True)
    font_files.sort()
    if only_basename:
        for file in font_files:
            file = os.path.splitext(os.path.basename(file))[0]
    return font_files

def generate_samples(font_family):
    text_length = random.randint(7, 18)
    text = "".join(random.choice(config.characters) for _ in range(text_length))

    # the images should have a size of 700px x 150px
    image_size = (700, 150)
    font_size = 70

    font = ImageFont.truetype(font_family, font_size)

    # noise = np.random.randint(0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
    noise = 255 * np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) 
    image = Image.fromarray(noise, "RGB")
    draw = ImageDraw.Draw(image)

    x = random.randint(0, abs(image_size[0] - font_size * text_length))
    y = random.randint(0, image_size[1] - font_size)

    draw.text((x, y), text, fill="black", font=font)
    return image

def generate_data(sample_number):
    font_paths = load_fonts()

    if not os.path.exists("data"):
        os.makedirs("data")
    for font_path in font_paths:
        fontname = os.path.splitext(os.path.basename(font_path))[0]

        if not os.path.exists("data/" + fontname):
            os.makedirs("data/" + fontname)

        for i in range(int(sample_number)):
            image = generate_samples(font_path)
            image.save("data/" + fontname + "/" + str(i) + ".jpg")

def load_data():
    """
    Loads all paths
    :return:
    """
    dir_path = os.getcwd()
    jpg_files = glob.glob(os.path.join(dir_path, 'data/**/*.jpg'), recursive=True)
    jpg_files.sort()
    return jpg_files

class GoogleFontsClassifier(nn.Module):
    def __init__(self, output_classes, dropout=0.5):
        super(GoogleFontsClassifier, self).__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1")

        self.resnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.avgpool,
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, H, W)
        x = self.resnet(x)
        return x


def train():
    # Load dataset
    total_dataset = Datagenerator(config)
    config.output_classes = len(total_dataset.fonts_unique)
    print(f"Total unique fonts: {config.output_classes}")

    model = GoogleFontsClassifier(config.output_classes, dropout=config.dropout)
    model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0.00001)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"model/model_{timestamp}.pth"

    # Split dataset into training, validation, and test set randomly
    indices = np.arange(len(total_dataset))
    np.random.shuffle(indices)
    split = int(len(indices) * (4 / 5))

    train_dataset = Subset(total_dataset, indices=indices[:split])
    val_dataset = Subset(total_dataset, indices=indices[split:])

    # Create datasets and dataloaders with rotated targets without augmentation (for evaluation)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    wandb.init(
        project="cantaffordthatfont",
        config={
            "learning_rate": config.learning_rate,
            "architecture": config.wandb_tag,
            "batch_size": config.batch_size,
            "epochs": config.num_epochs,
            "device": config.device,
        }
    )

    best_test_accuracy = -np.inf

    for epoch in range(config.num_epochs):
        correct_n = 0
        correct_top4 = 0

        # Train the model
        model.train()
        scaler = torch.amp.GradScaler()
        for inputs, targets in tqdm(train_dataloader, desc=f"Training (Epoch {epoch+1}/{config.num_epochs})", leave=False):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            correct_n += torch.sum(torch.argmax(outputs, dim=1) == targets).item()
            
            # Compute top-4 accuracy
            top4 = torch.topk(outputs, k=4, dim=1).indices  # (batch_size, 4)
            targets_expanded = targets.view(-1, 1).expand_as(top4)
            correct_top4 += torch.sum(top4 == targets_expanded).item()

            wandb.log({"training_loss": loss.item()})
        wandb.log({"train_accuracy": correct_n / len(train_dataset)})
        wandb.log({"train_accuracy_top4": correct_top4 / len(train_dataset)})

        # Evaluate the model on the validation set
        val_loss = 0.0
        correct_n = 0
        correct_top4 = 0

        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc=f"Validation (Epoch {epoch+1}/{config.num_epochs})", leave=False):
                inputs, targets = inputs.to(config.device), targets.to(config.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                correct_n += torch.sum(torch.argmax(outputs, dim=1) == targets).item()

                # Compute top-4 accuracy
                top4 = torch.topk(outputs, k=4, dim=1).indices  # (batch_size, 4)
                targets_expanded = targets.view(-1, 1).expand_as(top4)
                correct_top4 += torch.sum(top4 == targets_expanded).item()

        wandb.log({"validation_loss": val_loss / len(val_dataloader)})
        wandb.log({"validation_accuracy": correct_n / len(val_dataset)})
        wandb.log({"validation_accuracy_top4": correct_top4 / len(val_dataset)})

        if best_test_accuracy < correct_n / len(val_dataset):
            best_test_accuracy = correct_n / len(val_dataset)
            os.makedirs("model", exist_ok=True)
            torch.save(model.state_dict(), model_filename)

        scheduler.step()
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})

    wandb.finish()

if __name__ == "__main__":
    if config.generate_data:
        print("Generating data...")
        generate_data(sample_number=1000)
        print("Data generation complete.")

    train()
