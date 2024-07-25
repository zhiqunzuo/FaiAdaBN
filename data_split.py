import h5py
import pandas as pd
import numpy as np
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

def images_compressed(annot_data):
    images = []
    for path in annot_data["Path"]:
        image = cv2.imread(path)
        images.append(image)
    return images

def split_811(all_meta, patient_ids, seed):
    sub_train, sub_val_test = train_test_split(patient_ids, test_size=0.2, random_state=seed)
    sub_val, sub_test = train_test_split(sub_val_test, test_size=0.5, random_state=seed)
    train_meta = all_meta[all_meta.md5hash.isin(sub_train)]
    val_meta = all_meta[all_meta.md5hash.isin(sub_val)]
    test_meta = all_meta[all_meta.md5hash.isin(sub_test)]
    
    os.makedirs("dataset/csvs/random_seed={}".format(seed), exist_ok=True)
    
    train_meta.to_csv("dataset/csvs/random_seed={}/train.csv".format(seed))
    val_meta.to_csv("dataset/csvs/random_seed={}/val.csv".format(seed))
    test_meta.to_csv("dataset/csvs/random_seed={}/test.csv".format(seed))
    
    train_images = images_compressed(train_meta)
    val_images = images_compressed(val_meta)
    test_images = images_compressed(test_meta)
    
    os.makedirs("dataset/pkls/random_seed={}".format(seed), exist_ok=True)
    
    with open("dataset/pkls/random_seed={}/train_images.pkl".format(seed), "wb") as f:
        pickle.dump(train_images, f)
    with open("dataset/pkls/random_seed={}/val_images.pkl".format(seed), "wb") as f:
        pickle.dump(val_images, f)
    with open("dataset/pkls/random_seed={}/test_images.pkl".format(seed), "wb") as f:
        pickle.dump(test_images, f)

def data_spliting(seed):
    # read metadata
    #path = 'fitz17kdata'
    path = "data"

    annot_data = pd.read_csv(path + '/fitz17k.csv')

    pathlist = annot_data['md5hash'].values.tolist()
    paths = [path + '/' + 'finalfitz17k/' + i + '.jpg' for i in pathlist]

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
    
    split_811(annot_data, np.unique(annot_data['md5hash']), seed)