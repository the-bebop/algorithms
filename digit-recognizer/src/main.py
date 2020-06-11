import os
import acquisition as acq
import numpy as np
import random
import classifier

def get_config() -> dict:    
    """
    Control center for the algorithm.

    Returns:
        Algorithm config as dictionary.
    """
    config = {}

    ## paths
    # train:                 path to the csv of the MNIST's training database.
    # test:                 path to the csv of the MNIST's testing database.
    # eval:                 path to the csv of the MNIST's evaluation database.
    cur_path = os.path.dirname(__file__)
    data_path = os.path.join(cur_path, "..", "data")
    train_path = os.path.join(data_path,"train","train.csv") 
    eval_path = os.path.join(data_path,"eval","sample_submission.csv")
    test_path = os.path.join(data_path,"test","test.csv")
    path_config = {"paths": {"train": train_path, "test": test_path, "eval": eval_path}}
    config.update(path_config)

    ## verbose
    # enabled: boolean      flag that shows further data samples
    # sample_timeout:       the time it takes to auto close the window, set to "0" manual window exiting. 
    verbose_config = {"verbose":{"enabled": False, "sample_timeout":3000}}
    config.update(verbose_config)

    ## algorithm
    # trainset_percentag:   percentag that defines how many of the first sample shall be used for the trainset, the remaining are used for the testset. No evalset is generated.
    # random_seed:          the value that is generated when working with random generators.
    training_config = {"algorithm": {"trainset_percentage": 0.75, "random_seed": 312}}
    config.update(training_config)

    return config

def show_code_in_action(test_data, algo):
    """
    Shows a random subset from the input data with the algorithm's estimate next to the grount information

    Args:
        test_data: the dataset where the subset should be created from: np.array([labelset, dataset])
    Returns:
        Algorithm config as dictionary.
    """
    if test_data is None:
        raise ValueError("Don't know from where to take the samples from.")
    if algo is None or algo.classifier is None:
        raise RuntimeError("Algorithm has not been trained.")

    inspection_set = []
    labels = []
    no_of_samples = 10
    for nth_draw in range(no_of_samples):
        sample = random.randint(0, len(test_data[0]))
        inspection_set.append(test_data[0][sample])
        labels.append(test_data[1][sample])

    estimates = algo.test(np.array(inspection_set))

    for sample in range(no_of_samples):
        window_title = "Estimated as '{}' gt-label '{}'".format(estimates[sample], labels[sample])
        acq.display_img(inspection_set[sample], window_title)

def main():
    config = get_config()
    random.seed(config["algorithm"]["random_seed"])
    train_data, test_data = acq.extract_data(config)

    if config["verbose"]["enabled"]:
        acq.show_data(train_data[0], config["verbose"]["sample_timeout"])

    algo = classifier.Classifier(config["algorithm"]["random_seed"])
    algo.train(train_data[0], train_data[1])
    
    estimates = algo.test(test_data[0])
    algo.eval(estimates, test_data[1])

    show_code_in_action(test_data, algo)
    print("Finished algorithm.")

if __name__ == "__main__":
    main()