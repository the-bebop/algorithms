
import cv2

def corner_signature(x):
    return cv2.cornerHarris(x, 2, 5, 0.07) 

    