import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import f1_score

import os

data_dir = 'data/celeba'
wanted_attr = 'Blond_Hair'
train_max_index = 162770
val_max_index = 182637
test_max_index = 202599

def split_dataset(root_path, num_samples):
    img_dir = os.path.join(data_dir, 'images')
    split_hint_path = os.path.join(data_dir, 'list_eval_partition.txt')
    attr_path = os.path.join(data_dir, 'list_attr_celeba.txt')

    df_partition = pd.read_csv(split_hint_path)

    df_attr = pd.read_csv(attr_path)
    df_attr.set_index('image_id', inplace=True)
    df_attr.replace(to_replace=-1, value=0, inplace=True)

    # Join partition and the attributes
    df_partition.set_index('image_id', inplace=True)
    df_par_attr = df_partition.join(df_attr[wanted_attr], how="inner")


    



def load_reshape_img(fname):
    img = cv2.imread(fname)
    # img = load_img(fname)
    x = img / 255.
    x = x.reshape((1,) + x.shape)
    return x

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def generate_df(df_par_attr, partition, attr, num_samples):
    img_dir = os.path.join(data_dir, 'images')
    img_width = 178
    img_height = 218

    df_ = df_par_attr[(df_par_attr['partition'] == partition) 
                           & (df_par_attr[attr] == 0)].sample(int(num_samples/2))
    df_ = pd.concat([df_,
                      df_par_attr[(df_par_attr['partition'] == partition) 
                                  & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(os.path.join(img_dir, fname)) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = df_[attr]
        
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(os.path.join(img_dir, index))
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (img_width, img_height)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_