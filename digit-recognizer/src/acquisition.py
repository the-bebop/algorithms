import cv2
import math
import random
import numpy as np
import pandas as pd

import processing

def extract_data(csv_path, dataset_folds):

    if dataset_folds <= 0 or dataset_folds >1:
        raise ValueError("dataset_folds needs to be within 0 ... 1")
    cur_asset = pd.read_csv(csv_path)
    labels = cur_asset["label"]
    data = cur_asset.drop("label", axis=1)
    
    no_of_samples = len(labels)
    end_of_train_samples = int(no_of_samples * dataset_folds)

    train_labels =  np.array(   labels[0:end_of_train_samples]              )
    train_set =     np.array(   data[0:end_of_train_samples], dtype=np.uint8)        
    test_labels =   np.array(   labels[end_of_train_samples:]               )
    test_set =      np.array(   data[end_of_train_samples:],  dtype=np.uint8) 

    return [train_set, train_labels], [test_set, test_labels]


def convert_to_img(x):
    x = np.array(x)
    one_dim = int(math.sqrt(x.shape[0]))
    return x.reshape((one_dim, one_dim))

def display_img(sample, window_title="Sample", timeout=0):
    if len(np.shape(sample)) ==1:
        sample = convert_to_img(sample)
    
    window_title = str(window_title)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 320,320)
    cv2.imshow(window_title, sample)

    cv2.waitKey(timeout)

def show_data(X, timeout):
    cur_x = random.choice(X)
    digit_img = convert_to_img(cur_x)
    corner_img = processing.corner_signature(digit_img)
    digit_img = np.hstack((corner_img, digit_img))

    # displaying
    display_img(digit_img, "Arbitrarily selected sample", timeout)
