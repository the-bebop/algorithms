import cv2
import math
import random
import numpy as np
import pandas as pd

import processing

def extract_data(config=None):
    """
    Extracts the MNIST data and converts it into processible data types. Also splits that database according to the specified percentage.

    Args:
        config: the program's common config
    Raises:
        IOError
        ValueError 
    Returns:
        [numpy 2D array with training samples, numpy 1D array with training labels, numpy 2D array with testing samples, numpy 2D array with testing labels.]
    """

    if config is None:
        raise ValueError("Missing 'config'. Cannot work without settings.")
    csv_path = config["paths"]["train"]
    trainset_percentage = config["algorithm"]["trainset_percentage"]

    if csv_path is None:
        raise IOError("No path to MNIST databse specified.")
    if trainset_percentage <= 0 or trainset_percentage >1:
        raise ValueError("Relation between train- and testset samples needs to be within 0 ... 1.")
    cur_asset = pd.read_csv(csv_path)
    labels = cur_asset["label"]
    data = cur_asset.drop("label", axis=1)
    
    end_of_train_samples = int(len(labels) * trainset_percentage)

    train_labels =  np.array(   labels[0:end_of_train_samples]              )
    train_set =     np.array(   data[0:end_of_train_samples], dtype=np.uint8)        
    test_labels =   np.array(   labels[end_of_train_samples:]               )
    test_set =      np.array(   data[end_of_train_samples:],  dtype=np.uint8) 

    return [train_set, train_labels], [test_set, test_labels]


def convert_to_img(x=None):
    """
    This function converts a sample of the MNIST dataset to a opencv displayable format.

    Args:
        x:   The input image vector.

    Returns:
        A numpy 2darray 
    """
    x = np.array(x)
    one_dim = int(math.sqrt(x.shape[0]))
    return x.reshape((one_dim, one_dim))

def display_img(sample=None, window_title="Sample", timeout=0):
    """
    Displays a given MNIST sample with the help of opencv.

    Args:
        sample:       the data that shall be visualized (if 1D array it will automatically be converted)
        window_title: the title of the visualization written on top of it.
        timeout:      the time when the window auto-closes and continues the program.
    """
    if len(np.shape(sample)) ==1:
        sample = convert_to_img(sample)
    
    window_title = str(window_title)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 320,320)
    cv2.imshow("./"+window_title+".png", sample)

    cv2.waitKey(timeout)

def show_data(X, timeout):
    """
    Helper function that randomly selects a sample of the database.
    This function is basically used to visualize a certain feature the get a feeling for its significance.

    Args:
        X:          the MNIST database as numpy 2darray
        timeout:    the time when the window auto-closes and continues the program.
    Returns:
        [numpy 2D array with training samples, numpy 1D array with training labels, numpy 2D array with testing samples, numpy 2D array with testing labels.]
    """
    cur_x = random.choice(X)
    digit_img = convert_to_img(cur_x)
    corner_img = processing.corner_signature(digit_img)
    digit_img = np.hstack((corner_img, digit_img))

    # displaying
    display_img(digit_img, "Arbitrarily selected sample", timeout)
