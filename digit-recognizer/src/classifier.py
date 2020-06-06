from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from time import time
import numpy as np

import processing
import acquisition

class Classifier:
    def __init__(self):
        self.classifier = None
    
    def extract_features(self, X=None):
        
        #TODO replace with logging
        print("- Start - Extracting features")
        start_time = time()
        feature_set = []
        for x in X:
            x = acquisition.convert_to_img(x) # so unnecessary
            feature_vector = []
            feature_vector.extend( processing.corner_signature(x).flatten() )
            feature_vector.extend( processing.part_occupancy(x, 7) )
            feature_vector.extend( x.flatten() )

            feature_set.append(np.asarray(feature_vector))
        
        #TODO replace with logging
        print("- End - Extracting features, took {} s".format(time() - start_time))
        return np.asarray(feature_set)

    def train(self, X=None, y=None):
        if X is None or y is None:
            raise ValueError("Missing Training information.")
        
        #TODO replace with logging
        print("- Start - Training")
        start_time = time()
        self.classifier = QDA()
        X = self.extract_features(X)
        self.classifier.fit(X, y)        
        #TODO replace with logging
        print("- End - Training, took {} s".format(time() - start_time))

    def test(self, X=None):
        if X is None:
            raise ValueError("Missing testing data")
        if self.classifier is None:
            raise ValueError("Won't work with a flaky classifier")
        
        #TODO replace with logging
        print("- Start - Testing")
        start_time = time()
        X = self.extract_features(X)
        #TODO replace with logging
        print("- End - Testing, took {} s".format(time() - start_time))
        return self.classifier.predict(X)

    def eval(self, estimates=None, ground_truth=None):
        if self.classifier is None:
            raise ValueError("Won't work with a flaky classifier")
        if not isinstance(estimates, np.ndarray) or len(estimates) < 1:
            raise ValueError("Won't work with a flaky estimates")
        if not isinstance(ground_truth, np.ndarray) or len(ground_truth) < 1:
            raise ValueError("Won't work with a flaky ground truth")
        if len(ground_truth) != len(estimates):
            raise ValueError("Can't work with different size of ground truth ({}) and estimates ({}). ".format(ground_truth.shape, estimates.shape))

        label_amount = len(self.classifier.classes_)
        confusion_matrix = np.zeros((label_amount, label_amount), dtype=np.int32)
        for cur_gt, cur_estimate in zip(ground_truth, estimates):
            confusion_matrix[cur_gt, cur_estimate] += 1

        print(" ------ estimates ------")
        for row in confusion_matrix:
            for col in row:
                print(col, "\t\t ", end="")
            print(" |")
        print(" -----------------------")
        
