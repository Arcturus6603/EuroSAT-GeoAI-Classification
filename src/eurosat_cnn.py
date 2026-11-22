#CNN
#Python3.10 + PyTorch cu128 + scikit-learn


import os
import random
import sys
from pathlib import Path


#Set GDAL_DATA
env_root = Path(sys.prefix)
gdal_xsd_files = list(env_root.rglob("gdalvrt.xsd"))
if gdal_xsd_files:
    os.environ['GDAL_DATA'] = str(gdal_xsd_files[0].parent)
    print(f"GDAL_DATA set to: {gdal_xsd_files[0].parent}")
else:
    print("GDAL data directory not find")


import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,precision_score,recall_score
import seaborn as sns
import time
import pandas as pd

#Resolve Chinese character encoding issues
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


#Fix random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


#Configure data path
DATA_ROOT = Path(r"D:\data_storage\geo_data\remote_sensed_data\EuroSAT\raw\EuroSATallBands\ds\images\remote_sensing\otherDatasets\sentinel_2\tif")


CLASSES = sorted([
    "AnnualCrop","Forest","HerbaceousVegetation","Highway",
    "Industrial","Pasture","PermanentCrop","Residential",
    "River","seaLake"
])


#Automatically detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))


#Custom Dataset Class
class EuroSATDataset(Dataset):
    def __init__(self,file_paths,labels,is_train=False):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        #Read image
        with rasterio.open(self.file_paths[idx]) as src:
            img = src.read().astype(np.float32)

        img = img / 10000.0
        '''
        #Data Augmentation
        if self.is_train:
            if np.random.rand() > 0.5:
                img = np.flip(img,axis=2).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img,axis=1).copy()
        '''
        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, label_tensor


#Get all file paths and tags
print("Scanning dataset...")
all_paths,all_labels = [],[]
for label_idx,cls in enumerate(CLASSES):
    cls_dir = DATA_ROOT / cls
    for tif_path in cls_dir.glob("*.tif"):
        all_paths.append(tif_path)
        all_labels.append(label_idx)


#Split train/test set
x_train_paths,x_test_paths,y_train,y_test = train_test_split(
    all_paths,all_labels,test_size=0.2,random_state=42,stratify=all_labels
)

train_dataset = EuroSATDataset(x_train_paths,y_train,is_train=True)
test_dataset = EuroSATDataset(x_test_paths,y_test,is_train=False)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=0)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=0)


#Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleCNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=13,out_channels=32,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8,256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#Train Cycle
set_seed(512)

EPOCHS = 20
print(f"\nStarting CNN training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for inputs,labels in train_loader:
        input,labels = inputs.to(device),labels.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - acc:{train_acc:.2f}% - Time: {time.time() - start_time:.1f}s")


#torch.save(model.state_dict(), "../models/test/eurosat_cnn_weights.pth")
#print("Model weights have been saved.")


'''
model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("../models/eurosat_cnn_weights.pth"))
model.eval()
'''


#Evaluate
print("\nStarting test set evaluation...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs,labels in test_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_targets = np.array(all_targets).flatten()


#Output accuracy and report
acc = accuracy_score(all_targets,all_preds)
print(f"Overall Accuracy: {acc:.4f}")
print(classification_report(all_targets,all_preds,target_names=CLASSES,digits=4))


#Confusion Matrix
cm = confusion_matrix(all_targets,all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt="g",cmap="Greens",
            xticklabels=CLASSES,
            yticklabels=CLASSES )
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title(f"EuroSAT CNN Confusion Matrix(OA={acc:.4f})")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
os.makedirs("../test",exist_ok=True)
plt.savefig("../test/no_aug/more512/cnn_confusion_matrix.png",dpi=150)
plt.show()


#F1
f1_scores = f1_score(all_targets,all_preds,average=None)
macro_f1 = f1_score(all_targets,all_preds,average="macro")

df_report = pd.DataFrame({
    'Classes': CLASSES,
    'F1 Score': f1_scores
})

df_report.to_csv("../test/no_aug/more512/cnn_f1_scores.csv",index=False)
plt.figure(figsize=(12,5))
plt.bar(CLASSES,f1_scores,color="green",linestyle="--",label=f"Macro F1 = {macro_f1:.4f}")
plt.xticks(rotation=45,ha="right")
plt.ylabel("F1-Score")
plt.title("CNN F1-Score per Class")
plt.legend()
plt.tight_layout()
plt.savefig("../test/no_aug/more512/cnn_f1_scores.png",dpi=150)
plt.show()