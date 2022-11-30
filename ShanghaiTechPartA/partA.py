import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from scipy.io import loadmat
from torchvision.io import read_image


class PartADataset(Dataset):
    def __init__(self, img_dir, cord_dir, transform=None, guassian=None):
        self.img_dir = img_dir
        self.cord_dir = cord_dir
        self.transform = transform
        self.len = len(os.listdir(img_dir))
        self.guassian = guassian
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "IMG_{}.jpg".format(idx+1))
        cord_path = os.path.join(self.cord_dir, "GT_IMG_{}.mat".format(idx+1))
        
        img = read_image(img_path)
        cord = loadmat(cord_path)['image_info'][0][0][0][0][0]
        cord = self.mark(img, cord)
        
        if self.transform:
            img = self.transform(img)
            cord = torch.from_numpy(cord)
            cord = self.transform(cord)
            cord = transforms.Lambda(lambda x: torch.where(x > 0, 1.0, 0.0))(cord)
        
        if self.guassian:
            cord = self.guassian(cord)
                        
        return img, cord
    
    def mark(self, img, cord):
        l = np.zeros(img.shape[1:])
        
        for x, y in cord:
            x = int(x)
            y = int(y)
            l[y, x] = 1
        
        l = l.reshape(1, *l.shape)
        return l

if __name__ == "__main__":
    # Define transforms
    transform_d = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])
    
    # Load data
    train_data = PartADataset("./part_A/train_data/images", "./part_A/train_data/ground-truth", transform=transform_d)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    
    # Test
    for i, data in enumerate(train_loader):
        print(data[0].shape)
        print(data[1].shape)
        break
    