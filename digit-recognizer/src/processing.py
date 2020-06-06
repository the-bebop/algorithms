
import cv2
import numpy as np
from statistics import median

def corner_signature(x=None, harris_block_size=3, harris_ksize=5, harris_k=0.025):
    if not isinstance(x, np.ndarray):
        raise ValueError("Wrong input param, expecting sample as numpy array.")
    return cv2.cornerHarris(x, harris_block_size, harris_ksize, harris_k) 

def part_occupancy(x, no_of_parts):    
    one_dim = x.shape[0]
    if one_dim % no_of_parts != 0:
        raise ValueError("Cannot separate data into equally sized subsets for 'part occupancy' feature. Revise parameters. Info: no_of_parts = {} image size = ({}, {})".format(no_of_parts, one_dim, one_dim))
    if not isinstance(x, np.ndarray):
        raise ValueError("Wrong input param, expecting sample as numpy array.")

    step = int(one_dim / no_of_parts)
    flat_sample = x.flatten()
    return [median(flat_sample[pixel:pixel + step]) for pixel in range(0,one_dim*one_dim, step)]




    