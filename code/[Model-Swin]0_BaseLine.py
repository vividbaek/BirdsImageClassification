#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from IPython.display import clear_output
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SwinForImageClassification, AutoFeatureExtractor
from torch.utils.data import DataLoader
import logging

# 로깅 설정
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 25
VERSION = "v1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED) 

df_train = pd.read_csv('./train.csv')
df_train['img_path'] = "./train/" + df_train['img_path'].str.split('/').str[-1]

X = df_train['img_path'].to_numpy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_train['label'].to_numpy())

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image=np.array(image))['image']

        if self.labels is not None:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
        return image, None

aug_transforms = A.Compose([
    A.OneOf([
        A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.6, 1.0)),  # Zoom in
        A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(1.0, 1.4)),  # Zoom out
    ], p=1.0),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


def create_data_loaders(X_train, y_train, X_valid, y_valid, X_test, batch_size):
    train_dataset = ImageDataset(X_train, y_train, transform=aug_transforms)
    valid_dataset = ImageDataset(X_valid, y_valid, transform=aug_transforms)
    test_dataset = ImageDataset(X_test, labels=None, transform=aug_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

test_directory = "./test"
X_test = [os.path.join(test_directory, img) for img in os.listdir(test_directory)]

skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    train_loader, valid_loader, test_loader = create_data_loaders(X_train, y_train, X_valid, y_valid, X_test, BATCH_SIZE)

class CustomSwinTransformer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomSwinTransformer, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
        self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k", num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

model = CustomSwinTransformer(NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item(), accuracy=100. * (outputs.argmax(1) == labels).sum().item() / len(labels))

    logging.info(f'Training - Epoch: {epoch+1}, Loss: {total_loss / len(data_loader)}, Accuracy: {total_correct / len(data_loader.dataset)}')
    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)

def validate_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, total_correct = 0, 0
    progress_bar = tqdm(data_loader, desc='Validating', leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), accuracy=100. * (outputs.argmax(1) == labels).sum().item() / len(labels))

    logging.info(f'Validation - Epoch: {epoch+1}, Loss: {total_loss / len(data_loader)}, Accuracy: {total_correct / len(data_loader.dataset)}')
    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)

for epoch in tqdm(range(EPOCHS), desc='Epochs'):
    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
    valid_loss, valid_acc = validate_epoch(model, valid_loader, loss_fn, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc*100:.2f}%')

model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.argmax(1).tolist())

predictions = np.array(predictions).reshape(-1, 1)

df_submission = pd.DataFrame({'label': predictions.flatten()})
df_submission.to_csv(f'./{VERSION}_submission.csv', index=False)
