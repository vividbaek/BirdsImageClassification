import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SwinForImageClassification, AutoFeatureExtractor
import logging
import optuna
from torch.cuda.amp import GradScaler, autocast

NUM_CLASSES = 25
EPOCHS = 80
SEED = 42



def load_data():
    # CSV 파일을 불러오기
    df_train = pd.read_csv('./train.csv')
    # 이미지 파일 경로를 업데이트
    df_train['img_path'] = "./train/" + df_train['img_path'].str.split('/').str[-1]

    # LabelEncoder를 사용하여 레이블을 정수로 변환
    label_encoder = LabelEncoder()
    df_train['label'] = label_encoder.fit_transform(df_train['label'])

    # 이미지 경로와 레이블을 numpy 배열로 변환
    X = df_train['img_path'].to_numpy()
    y = df_train['label'].to_numpy()
    return X, y


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

class CustomSwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k", num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, x):
        return self.model(x).logits

def save_model(model, filename="best_model.pth"):
    torch.save(model.state_dict(), filename)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    total_loss, total_correct, batch_count = 0, 0, 0
    optimizer.zero_grad()

    for inputs, labels in tqdm(data_loader):
        with autocast():  # AMP 적용
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels) / accumulation_steps  # 그라디언트 어큐뮬레이션을 위해 손실을 나눔

        # 스케일된 그라디언트를 계산
        scaler.scale(loss).backward()
        batch_count += 1

        # 그라디언트 어큐뮬레이션
        if batch_count % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)

def validate_epoch(model, data_loader, loss_fn, device, scaler):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            with autocast():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)


def create_data_loaders(X, y, batch_size, transforms):
    skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        train_dataset = ImageDataset(X_train, y_train, transform=transforms)
        valid_dataset = ImageDataset(X_valid, y_valid, transform=transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    p_rotate = trial.suggest_float('p_rotate', 0, 1)
    p_flip = trial.suggest_float('p_flip', 0, 1)
    scaler = GradScaler()  # GradScaler 초기화

    transforms = A.Compose([
        A.RandomResizedCrop(height=256, width=256, scale=(0.6, 1.0)),
        A.Rotate(limit=20, p=p_rotate),
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=p_flip),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomSwinTransformer(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    X, y = load_data()
    train_loader, valid_loader = create_data_loaders(X, y, batch_size, transforms)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss, _ = train_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        val_loss, _ = validate_epoch(model, valid_loader, loss_fn, device, scaler)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f"best_model_trial_{trial.number}.pth")

    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best trial:", study.best_trial.params)
