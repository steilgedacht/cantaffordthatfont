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
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from dataloader import Datasubsets


characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

def load_fonts(only_basename=False):
    dir_path = os.getcwd()
    font_files = glob.glob(os.path.join(dir_path, 'fonts_subset/*.ttf'), recursive=True)
    font_files.sort()
    if only_basename:
        for file in font_files:
            file = os.path.splitext(os.path.basename(file))[0]
    return font_files

def generate_samples(font_family):
    text_length = random.randint(7, 18)
    text = "".join(random.choice(characters) for _ in range(text_length))

    # the images should have a size of 700px x 150px
    image_size = (700, 150)
    font_size = 70

    font = ImageFont.truetype(font_family, font_size)

    noise = np.random.randint(0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
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

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 7 * 41, 512)  # Adjust input size based on your input
        self.fc2 = nn.Linear(512, 95)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension (1 channel for grayscale)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 256 * 7 * 41)  # Adjust the size based on your input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    model = MyModel()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 25
    batch_size = 64
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    results = []

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"model/model_{timestamp}.pth"

    # Load dataset
    total_dataset = Datasubsets()

    # Split dataset into training, validation, and test set randomly
    train_dataset = Subset(total_dataset, indices=np.arange(int(len(total_dataset) * (3 / 5))))
    val_dataset = Subset(total_dataset, indices=np.arange(int(len(total_dataset) * (3 / 5)), len(total_dataset)))

    # Create datasets and dataloaders with rotated targets without augmentation (for evaluation)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    wandb.init(
        # set the wandb project where this run will be logged
        project="cantaffordthatfont",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "CNN",
            "batch_size": batch_size,
            "epochs": num_epochs,
            "device": device,
        }
    )

    for epoch in range(num_epochs):

        results.append({"best_test_accuracy" : 0.0, "train_loss": 0.0, "train_accuracy": 0.0, "train_correct": 0, "val_loss": 0.0, "val_accuracy": 0.0, "val_correct": 0 })

        # Train the model
        model.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            results[epoch]["train_correct"] += torch.sum(torch.argmax(outputs.data, dim=1) == torch.argmax(targets.data, dim=1)).item()
            wandb.log({"training_loss": loss.item()})

        results[epoch]["train_accuracy"] = results[epoch]["train_correct"] / len(train_dataset)
        wandb.log({"train_accuracy": results[epoch]["train_accuracy"]})

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                results[epoch]["val_loss"] += loss.item()
                results[epoch]["val_correct"] += torch.sum(torch.argmax(outputs.data, dim=1) == torch.argmax(targets.data, dim=1)).item()
                wandb.log({"validation_loss": loss.item()})
        results[epoch]["val_accuracy"] = results[epoch]["val_correct"] / len(val_dataset)
        wandb.log({"validation_accuracy": results[epoch]["val_accuracy"]})

        if results[epoch]["best_test_accuracy"] < results[epoch]["val_accuracy"]:
            results[epoch]["best_test_accuracy"] = results[epoch]["val_accuracy"]
            torch.save(model.state_dict(), model_filename)

    wandb.finish()

if __name__ == "__main__":
    # generate_data(sample_number=1e3)
    train()
