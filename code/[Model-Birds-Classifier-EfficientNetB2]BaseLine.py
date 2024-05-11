import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # Corrected import
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from transformers import EfficientNetForImageClassification, AutoFeatureExtractor
import optuna
from torch.cuda.amp import GradScaler, autocast

NUM_CLASSES = 25
EPOCHS = 10
SEED = 42

def load_data():
    df_train = pd.read_csv('./train.csv')
    df_train['img_path'] = "./train/" + df_train['img_path'].str.split('/').str[-1]
    label_encoder = LabelEncoder()
    df_train['label'] = label_encoder.fit_transform(df_train['label'])
    X = df_train['img_path'].to_numpy()
    y = df_train['label'].to_numpy()
    return X, y, label_encoder.classes_

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

def create_data_loaders(X, y, batch_size, transform):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    data_loaders = []
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        train_dataset = ImageDataset(X_train, y_train, transform=transform)
        valid_dataset = ImageDataset(X_valid, y_valid, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        data_loaders.append((train_loader, valid_loader))
    return data_loaders

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    img_size = trial.suggest_categorical('img_size', [224, 256, 299, 331, 384])
    p_rotate = trial.suggest_float('p_rotate', 0.0, 1.0)
    p_flip = trial.suggest_float('p_flip', 0.0, 1.0)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=20, p=p_rotate),
        A.HorizontalFlip(p=p_flip),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    X, y, _ = load_data()
    loaders = create_data_loaders(X, y, batch_size, transform)
    train_loader, valid_loader = loaders[0]  # Use first split for simplicity

    model = EfficientNetForImageClassification.from_pretrained(
        "dennisjooo/Birds-Classifier-EfficientNetB2",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        model.train()
        for images, labels in tqdm(train_loader, desc="Training Batch", leave=False):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images).logits
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        model.eval()
        total_loss = 0
        for images, labels in tqdm(valid_loader, desc="Validation Batch", leave=False):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images).logits
                loss = loss_fn(outputs, labels)
            total_loss += loss.item()

        avg_val_loss = total_loss / len(valid_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.4f}")
    
    return best_val_loss

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    print("Best trial:", study.best_trial.params)
