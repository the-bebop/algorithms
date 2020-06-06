import cv2
import math
import random
import numpy as np
import pandas as pd

import processing

def extract_data(csv_path):

    cur_asset = pd.read_csv(csv_path)
    labels = cur_asset["label"]
    data = cur_asset.drop("label", axis=1)
    
    return np.array(labels), np.array(data, dtype=np.uint8)

def convert_to_img(x):
    x = np.array(x)
    one_dim = int(math.sqrt(x.shape[0]))
    return x.reshape((one_dim, one_dim))

def show_data(X, timeout):
    cur_x = random.choice(X)
    digit_img = convert_to_img(cur_x)
    corner_img = processing.corner_signature(digit_img)
    digit_img = np.hstack((corner_img, digit_img))

    # displaying
    win_title = "Arbitrarily selected sample"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 320,320)
    cv2.imshow(win_title, digit_img)
    cv2.waitKey(timeout)
