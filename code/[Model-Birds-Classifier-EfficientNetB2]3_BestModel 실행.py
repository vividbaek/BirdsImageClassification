import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import EfficientNetForImageClassification
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib  # LabelEncoder 저장 및 로딩에 사용

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 10,
    'LEARNING_RATE': 8.097618657535927e-05,
    'BATCH_SIZE': 32,
    'SEED': 42,
    'p_rotate': 0.2497816051709541,
    'p_flip': 0.4050026202988107,
    'MODEL_PATH': 'efficientnet_best.pth'  # 최적의 모델 저장 경로
}

# Set random seeds for reproducibility
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG['SEED'])

# Prepare dataset
df = pd.read_csv('./train.csv')
train, val = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])

# Label encoding
le = LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])
joblib.dump(le, 'label_encoder.joblib')  # LabelEncoder 저장

# Custom Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['img_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        label = -1 if self.test else self.df.iloc[index]['label']
        return image, label

# Transformations
def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
            A.Rotate(limit=45, p=CFG['p_rotate']),
            A.HorizontalFlip(p=CFG['p_flip']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# Loaders
train_dataset = CustomDataset(train, transforms=get_transforms(data='train'))
val_dataset = CustomDataset(val, transforms=get_transforms(data='valid'))
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# Model initialization
model = EfficientNetForImageClassification.from_pretrained(
    "google/efficientnet-b3",
    num_labels=len(le.classes_),
    ignore_mismatched_sizes=True
).to(device)

# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Training and Validation functions
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    best_loss = float('inf')
    for images, labels in tqdm(val_loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), CFG['MODEL_PATH'])  # 최적의 모델 저장
    return avg_loss

# Train and validate the model
criterion = nn.CrossEntropyLoss()
for epoch in range(CFG['EPOCHS']):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Assuming test data is available and prepared similarly
test_df = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test_df, transforms=get_transforms(data='valid'), test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# Load best model
model.load_state_dict(torch.load(CFG['MODEL_PATH']))
model.to(device)

# Inference function
def inference(model, test_loader, device, label_encoder):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Running Inference"):
            images = images.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
    return label_encoder.inverse_transform(preds)  # Apply inverse transform to get original labels

# Load LabelEncoder
le = joblib.load('label_encoder.joblib')

# Perform inference
preds = inference(model, test_loader, device, le)

# Generate submission file
submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds
submit.to_csv('./final_submission.csv', index=False)
