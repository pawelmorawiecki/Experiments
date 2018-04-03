import numpy as np
import h5py
import os

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
    
    sensitivity = true_pos / (false_neg + true_pos)
    specificity = true_neg / (true_neg + false_pos)
    return (sensitivity+specificity)/2