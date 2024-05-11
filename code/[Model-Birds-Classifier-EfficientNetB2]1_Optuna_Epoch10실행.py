import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from transformers import EfficientNetForImageClassification
import optuna
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

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
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(skf.split(X, y))
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    train_dataset = ImageDataset(X_train, y_train, transform=transform)
    valid_dataset = ImageDataset(X_valid, y_valid, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

def train_and_validate(train_loader, valid_loader, model, optimizer, loss_fn, device, epochs, params):
    scaler = GradScaler()
    print(f"Training with parameters: {params}")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(images).logits
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        model.eval()
        total_loss = 0
        with tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    with autocast():
                        outputs = model(images).logits
                        loss = loss_fn(outputs, labels)
                    total_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())

        avg_val_loss = total_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}: Average Training Loss = {total_train_loss / len(train_loader):.6f}, Validation Loss = {avg_val_loss:.6f}")
    return avg_val_loss  # 이제 마지막 에폭의 평균 검증 손실을 반환합니다.


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    img_size = trial.suggest_categorical('img_size', [224, 256, 299])
    p_rotate = trial.suggest_float('p_rotate', 0.0, 0.5)
    p_flip = trial.suggest_float('p_flip', 0.0, 0.5)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=20, p=p_rotate),
        A.HorizontalFlip(p=p_flip),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    X, y, _ = load_data()
    train_loader, valid_loader = create_data_loaders(X, y, batch_size, transform)

    # Use a valid model identifier
    model = EfficientNetForImageClassification.from_pretrained(
        "google/efficientnet-b3",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    return train_and_validate(train_loader, valid_loader, model, optimizer, loss_fn, device, EPOCHS, trial.params)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print("Best trial:", study.best_trial.params)

    best_params = study.best_trial.params
    transform = A.Compose([
        A.Resize(best_params['img_size'], best_params['img_size']),
        A.Rotate(limit=20, p=best_params['p_rotate']),
        A.HorizontalFlip(p=best_params['p_flip']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    X, y, _ = load_data()
    train_loader, valid_loader = create_data_loaders(X, y, best_params['batch_size'], transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetForImageClassification.from_pretrained(
        "dennisjooo/Birds-Classifier-EfficientNetB2",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()

    train_and_validate(train_loader, valid_loader, model, optimizer, loss_fn, device, EPOCHS, best_params)

if __name__ == "__main__":
    main()
