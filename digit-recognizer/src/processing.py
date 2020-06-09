
import cv2
import numpy as np
from statistics import median
from scipy.stats import pearsonr

def corner_signature(x=None, harris_block_size=3, harris_ksize=5, harris_k=0.025):
    """
    Feature that highlights the corner spots (uses Harris detection).

    Args:
        x:          sample to process
        harris_*:   see opencv description    
    Raises:
        ValueError 
    Returns:
        feature image
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Wrong input param, expecting sample as numpy array.")
    return cv2.cornerHarris(x, harris_block_size, harris_ksize, harris_k) 

def part_occupancy(x, no_of_parts):
    """
    Feature that subdivides the input sample into sectors and points out the occupancies (e.g. mean cell intensity) there.

    Args:
        x:              sample to process
        no_of_parts:    amount of sectors for fragmentation.
    Raises:
        ValueError 
    Returns:
        feature image
    """
    one_dim = x.shape[0]
    if one_dim % no_of_parts != 0:
        raise ValueError("Cannot separate data into equally sized subsets for 'part occupancy' feature. Revise parameters. Info: no_of_parts = {} image size = ({}, {})".format(no_of_parts, one_dim, one_dim))
    if not isinstance(x, np.ndarray):
        raise ValueError("Wrong input param, expecting sample as numpy array.")

    step = int(one_dim / no_of_parts)
    flat_sample = x.flatten()
    return [median(flat_sample[pixel:pixel + step]) for pixel in range(0,one_dim*one_dim, step)]

def class_correlation(x, comparables):
    """
    Feature that compares the input sample with a database of class appearances.

    Note that the "comparables" are generated in Classifier::get_comparables().

    Args:
        x:              sample to process
        comparables:    dictionary containing one correlation sample per class.
    Returns:
        [correlation result, p_value]
    """
    ret_vec = []

    for key in comparables.keys():
        comparable = comparables[key]
        corr, p_value = pearsonr(x.flatten(), comparable)
        ret_vec.append(corr)
        ret_vec.append(p_value)

    return ret_vec



    