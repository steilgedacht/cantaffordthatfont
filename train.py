import numpy as np
import os
import glob
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
    dir_path = os.getcwd()
    jpg_files = glob.glob(os.path.join(dir_path, 'data/**/*.jpg'), recursive=True)
    jpg_files.sort()

    jpg_file_label = []
    for jpg_file in jpg_files:
        pass
    return jpg_files

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding="same")
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding="same")
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, padding="same")
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding="same")
        self.bn4 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=15, padding="same")
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, padding="same")
        self.bn6 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=6)
        self.fc1 = nn.Linear(128*11, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn8 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self, x):
        x = self.pool1(self.bn2(self.relu(self.conv2(self.bn1(self.relu(self.conv1(x)))))))
        x = self.pool2(self.bn4(self.relu(self.conv4(self.bn3(self.relu(self.conv3(x)))))))
        x = self.pool3(self.bn6(self.relu(self.conv6(self.bn5(self.relu(self.conv5(x)))))))
        x = x.view(-1, 128*11)
        x = self.dropout(self.relu(self.bn7(self.fc1(x))))
        x = self.dropout(self.relu(self.bn8(self.fc2(x))))
        x = self.fc3(x)
        return x


def train():
    X = load_data()

    model = MyModel()

    num_epochs = 25
    batch_size = 64
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    test_accuracy = 0
    training_accuracy = 0

    results = []

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"model_{timestamp}.pth"

    train_dataset = TensorDataset(torch.from_numpy(X[24000:]).float(), torch.from_numpy(y[24000:]).float())
    val_dataset =   TensorDataset(torch.from_numpy(X[:24000]).float(), torch.from_numpy(y[:24000]).float())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):

        results = {"best_test_accuracy" : 0.0, "train_loss": 0.0, "train_accuracy": 0.0, "train_correct": 0, "val_loss": 0.0, "val_accuracy": 0.0, "val_correct": 0 }

        # Train the model
        model.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            results[epoch]["train_loss"] += loss.item() 
            results[epoch]["train_correct"] += (torch.argmax(outputs.data, dim=1) == torch.argmax(targets.data, dim=1)).sum().item()
        results[epoch]["train_accuracy"] = results[epoch]["train_correct"] / len(train_dataset)


        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs, targets)

                results[epoch]["val_loss"] += loss.item() 
                results[epoch]["val_correct"] += (torch.argmax(outputs.data, dim=1) == torch.argmax(targets.data, dim=1)).sum().item()
        results[epoch]["val_accuracy"] = results[epoch]["val_correct"] / len(val_dataset)

if __name__ == "__main__":
    # generate_data(sample_number=1e3)
    train()
