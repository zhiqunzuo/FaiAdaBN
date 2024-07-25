import h5py
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

# read metadata
path = 'fitz17kdata'

annot_data = pd.read_csv(path + '/fitz17k.csv')

pathlist = annot_data['md5hash'].values.tolist()
paths = ['images/' + i + '.jpg' for i in pathlist]

annot_data['Path'] = paths

# remove skin type == null 
annot_data = annot_data[annot_data['fitzpatrick_scale'] != -1]

# binarize the label
labellist = annot_data['three_partition_label'].values.tolist()
labels = [1 if x == 'malignant' else 0 for x in labellist]
print(Counter(labels))
annot_data['binary_label'] = labels

annot_data['skin_type'] = annot_data['fitzpatrick_scale'] - 1
skin_lists = annot_data['skin_type'].values.tolist()
annot_data['skin_binary'] = [0 if x <=2 else 1 for x in skin_lists] 

def split_811(all_meta, patient_ids):
    sub_train, sub_val_test = train_test_split(patient_ids, test_size=0.2, random_state=5)
    sub_val, sub_test = train_test_split(sub_val_test, test_size=0.5, random_state=6)
    train_meta = all_meta[all_meta.md5hash.isin(sub_train)]
    val_meta = all_meta[all_meta.md5hash.isin(sub_val)]
    test_meta = all_meta[all_meta.md5hash.isin(sub_test)]
    return train_meta, val_meta, test_meta

sub_train, sub_val, sub_test = split_811(annot_data, np.unique(annot_data['md5hash']))

sub_train.to_csv('fitz17kdata/split/new_train.csv')
sub_val.to_csv('fitz17kdata/split/new_val.csv')
sub_test.to_csv('fitz17kdata/split/new_test.csv')