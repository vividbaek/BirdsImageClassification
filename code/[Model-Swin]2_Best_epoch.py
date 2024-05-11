#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install --user albumentations


# ## Import

# In[2]:


import random
import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import SwinForImageClassification
import warnings
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
warnings.filterwarnings(action='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)


# ## Hyperparameter Settting

# In[4]:


CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 80,
    'LEARNING_RATE': 2.1895923818665253e-05,  # 하이퍼파라미터 튜닝 결과 기반 설정
    'BATCH_SIZE': 32,
    'SEED': 42,
    'p_rotate': 0.4153718538408363,
    'p_flip' : 0.25725408435593533
}


# ## Fixed RandomSeed

# In[5]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ## Train & Validation Split

# In[6]:


df = pd.read_csv('./train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])


# ## Label-Encoding

# In[7]:


le = preprocessing.LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])


# ## CustomDataset

# In[8]:


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        label = self.label_list[index] if self.label_list is not None else -1
        return image, label
    
    def __len__(self):
        return len(self.img_path_list)

# Augmentation including rotations and flips
train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Rotate(limit=45, p=CFG['p_rotate']),  # Updated based on hyperparameter tuning
    A.HorizontalFlip(p=CFG['p_flip']),  # Updated based on hyperparameter tuning
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2()
])

train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


# ## Model Define

# In[9]:


class CustomSwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k", num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits  # 직접 logits를 반환


# In[10]:


def load_model(model_path, num_classes):
    # Swin Transformer 모델을 초기화하고 학습된 상태를 로드합니다.
    model = CustomSwinTransformer(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


# In[11]:


num_classes = len(le.classes_)
model_path = './best_model_trial_1.pth'
model = load_model(model_path, num_classes)

optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)


# In[12]:


num_classes = len(le.classes_)  # le는 LabelEncoder 인스턴스
model_path = './best_model_trial_1.pth'

model = load_model(model_path, num_classes)
optimizer = Adam(model.parameters(), lr=CFG["LEARNING_RATE"])
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)


# In[13]:


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0.0
    best_model = None

    for epoch in range(CFG['EPOCHS']):
        model.train()
        train_loss = []
        # tqdm을 사용하여 훈련 데이터 로더를 감싸 훈련 과정의 진행 상태를 나타냅니다.
        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            imgs = imgs.to(device).float()
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_loss, val_score = validation(model, criterion, val_loader, device)
        print(f'Epoch {epoch+1}, Train Loss: {np.mean(train_loss):.5f}, Val Loss: {val_loss:.5f}, Val F1 Score: {val_score:.5f}')
        scheduler.step(val_score)

        if val_score > best_score:
            best_score = val_score
            best_model = model

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    true_labels, preds = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating"):
            imgs = imgs.to(device).float()
            labels = labels.to(device).long()
            outputs = model(imgs)  # 이 부분을 변경
            loss = criterion(outputs, labels)  # 여기에서도 변경
            preds.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
            true_labels.extend(labels.detach().cpu().numpy().tolist())
            val_loss.append(loss.item())
    val_loss = np.mean(val_loss)
    val_score = f1_score(true_labels, preds, average='macro')
    return val_loss, val_score


# In[14]:


infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


# In[ ]:


def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _ in test_loader:  # 라벨이 없으므로, _를 사용하여 무시
            imgs = imgs.to(device).float()
            outputs = model(imgs)
            preds.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
    preds = le.inverse_transform(preds)
    return preds


# In[ ]:


preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds
submit.to_csv('./baseline_submit_05.07.csv', index=False)

