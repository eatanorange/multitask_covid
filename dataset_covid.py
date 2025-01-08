import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

root='dataset'

train_data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),    # 随机垂直翻转
    transforms.RandomRotation(10),      # 随机旋转，最大10度
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并缩放
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
    transforms.ToTensor()
     
])

val_data_transforms=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    
])


class mydataset(Dataset):
    def __init__(self, root,path, transform=None):
        filenames = []
        labels1 = []
        labels2 = []
        self.root = root
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split()  # 移除行尾的换行符并按空格分割
                filename = parts[0]
                label1 = int(parts[1])
                label2 = int(parts[2])
                filenames.append(filename)
                labels1.append(label1)
                labels2.append(label2)



        self.all_image_paths = filenames
        self.all_image_labels = labels1
        self.transform =transform

    def __getitem__(self, index):
        img = Image.open(self.root+'/covid/images/' + self.all_image_paths[index]).convert('RGB')
        img = self.transform(img)
        label = self.all_image_labels[index]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.all_image_paths)


train_path = 'dataset/covid/train.txt'
test_path = 'dataset/covid/test.txt'

covid_train_dataset = mydataset(root,train_path, train_data_transforms)
covid_val_dataset = mydataset(root,test_path, val_data_transforms)