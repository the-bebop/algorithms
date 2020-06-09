from sklearn.ensemble import RandomForestClassifier
from time import time
import numpy as np

import processing
import acquisition

class Classifier:
    def __init__(self):
        self.classifier = None
        self.class_samples = None
    
    def get_comparables(self, X=None, y=None):
        """
        Takes the database and generates one image per class with the overal appearance of its category. This is used for feature extraction.

        Args:
            X:  database to extract the image samples from.
            y:  labels for categorization.
            --> X & y sample relation is connected via their position (i.e. X[n] belongs to y[n], where n is an arbitrary existing sample ID)        
        Returns:
            [numpy 2D array with training samples, numpy 1D array with training labels, numpy 2D array with testing samples, numpy 2D array with testing labels.]
        """ 
        class_samples = {}
        # start with class independend container
        for cur_class in set(y):
            class_samples.update({cur_class: []})

        # get digit samples
        for cur_sample, cur_y in zip(X, y):
            class_samples[cur_y].append(cur_sample)

        # create average class appearance(s)
        for key in class_samples.keys():
            cur_class_stack = np.asarray(class_samples[key])
            comparable = [np.median(cur_class_stack[:, pixel]) for pixel in range(cur_class_stack[0].shape[0])]
            class_samples[key] = np.array(comparable)

        self.class_samples = class_samples

    def extract_features(self, X=None):
        """
        Represents the feature extraction pipeline for an identical training and testing.

        Args:
            X:  the raw dataset (train or test) to process
        Returns:
            The featureset related to the given input dataset
        """
        #TODO replace with logging
        print("- Start - Extracting features")
        start_time = time()
        feature_set = []
        for x in X:
            x = acquisition.convert_to_img(x) # so unnecessary
            feature_vector = []
            #feature_vector.extend( processing.corner_signature(x).flatten() )
            #feature_vector.extend( processing.part_occupancy(x, 7) )
            #feature_vector.extend( x.flatten() )
            feature_vector.extend(  processing.class_correlation(x, self.class_samples))

            feature_set.append(np.asarray(feature_vector))
        
        #TODO replace with logging
        print("- End - Extracting features, took {} s".format(time() - start_time))
        return np.asarray(feature_set)

    def train(self, X=None, y=None):
        """
        Represents the classifier's training pipeline.

        Args:
            X: the raw trainingset
            y: the related training labels
        Raises:
            ValueError 
        """
        if X is None or y is None:
            raise ValueError("Missing Training information.")
        
        #TODO replace with logging
        print("- Start - Training")
        start_time = time()
        x = self.get_comparables(X, y)
        self.classifier = RandomForestClassifier()
        X = self.extract_features(X)
        self.classifier.fit(X, y)        
        #TODO replace with logging
        print("- End - Training, took {} s".format(time() - start_time))

    def test(self, X=None):
        """
        Represents the classifier's training pipeline.

        Args:
            X: the raw testset
            y: the related training labels
        Raises:
            ValueError 
        Returns:
            the estimate_set related to the input dataset
        """
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
        """
        Represents the evaluation pipeline that compares the classifier's estimate with the actual ground truth.

        Prints:
        - confusion matrix
        - classification metrics

        Args:
            estimates:      the estimateset generated from Classifier::test()
            ground_truth:   the comparison labels from the dataset
        Raises:
            ValueError 
        """
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
        
        results = {"confusion_matrix": confusion_matrix, "classes": {}}
        for cur_class in range(label_amount):        
            TP = TN = FP = FN = 0
            TP = confusion_matrix[cur_class, cur_class]
            FP = sum(confusion_matrix[cur_class,:]) - TP
            FN = sum(confusion_matrix[:,cur_class]) - TP
            TN = sum(confusion_matrix) - TP - FP - FN
            class_results = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

            precision = class_results["TP"]/(class_results["TP"] + class_results["FP"])
            recall = precision = class_results["TP"]/(class_results["TP"] + class_results["FN"])
            metric_results = {"precision": precision, "recall": recall}
            results["classes"].update({cur_class: {"details": class_results, "metrics": metric_results}})


        # evaluation
        for cur_class in results["classes"].keys():
            print("Class {}: precision {}, recall {}.".format(  cur_class,
                                                                results["classes"][cur_class]["metrics"]["precision"], 
                                                                results["classes"][cur_class]["metrics"]["recall"])
                                                                )