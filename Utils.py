import numpy as np
import h5py
import os
from torch.utils.data import Dataset, DataLoader

def save_large_dataset(file_name, variable):
    h5f = h5py.File(file_name + '.h5', 'w')
    h5f.create_dataset('variable', data=variable)
    h5f.close()

def load_large_dataset(file_name):
    h5f = h5py.File(file_name + '.h5','r')
    variable = h5f['variable'][:]
    h5f.close()
    return variable

# Metric is the mean of sensitivity and specificity
def calculate_metric(predictions, labels):
    true_pos = 0.0
    false_pos = 0.0
    true_neg = 0.0
    false_neg = 0.0
    
    for i in range (len(predictions)):
        if (predictions[i]==1 and labels[i]==1): true_pos +=1
        if (predictions[i]==1 and labels[i]==0): false_pos +=1
        if (predictions[i]==0 and labels[i]==0): true_neg +=1
        if (predictions[i]==0 and labels[i]==1): false_neg +=1
    
    if (true_pos==0): sensitivity = 0 #to avoid division by zero 
    else: sensitivity = true_pos / (false_neg + true_pos)
  
    if (true_neg==0): specificity = 0 #to avoid division by zero
    else: specificity = true_neg / (true_neg + false_pos)
    
    return (sensitivity+specificity)/2


class PAC2018Dataset(Dataset):
    """PAC 2018 Competition dataset."""

    def __init__(self, image_features, extra_features, labels):
        self.image_features = image_features
        self.extra_features = extra_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.image_features[idx], self.extra_features[idx], self.labels[idx]
        return sample
    

class DatasetTransforms(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):        
        image = self.transform(self.images[idx])
        sample = image, self.labels[idx]
        return sample