from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
from sklearn.model_selection import train_test_split
from nava import play, stop
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {gpu}")
print(f"GPU Available: {torch.cuda.is_available()}")
full_dataset = datasets.ImageFolder(root=r"C:\Users\sbelu\Desktop\Projects\FruitID\Fruit Identifier\Datasets\Resized_Fruits-262\208x256")

targets = np.array(full_dataset.targets)
indices = np.arange(len(targets))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0,
    stratify=targets,
    random_state=999
)

class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label
        
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224),
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAdjustSharpness(0.7, p=0.15),
    transforms.RandomAdjustSharpness(1.3, p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = TransformSubset(full_dataset, train_idx, transform=train_transform)
test_dataset = TransformSubset(full_dataset, test_idx, transform=test_transform)


trl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

print(f"Train Size: {len(train_dataset)} images \nTest Size:  {len(test_dataset)} images")
start = time()
for i, (inputs, labels) in enumerate(trl):
    if i == 3000:
        break
print(f"{i} batches in {time() - start:.2f}s with 0 workers")