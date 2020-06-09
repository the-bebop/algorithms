import os
import acquisition as acq
import classifier

def get_config(with_verbose) -> dict:    
    """
    Control center for the algorithm.

    Args:
        with_verbose: set to true if further information should be viewed (e.g. show samples)
    Returns:
        Algorithm config as dictionary.
    """
    config = {}

    #paths
    cur_path = os.path.dirname(__file__)
    data_path = os.path.join(cur_path, "..", "data")
    train_path = os.path.join(data_path,"train","train.csv")
    eval_path = os.path.join(data_path,"eval","sample_submission.csv")
    test_path = os.path.join(data_path,"test","test.csv")
        
    path_config = {"paths": {"train": train_path, "test": test_path, "eval": eval_path}}
    config.update(path_config)

    #debug 
    verbose_config = {"verbose":{"enabled": with_verbose, "sample_timeout":3000}}
    config.update(verbose_config)

    #training
    training_config = {"training": {"dataset_folds": 0.75}}
    config.update(training_config)

    return config


def main():
    verbose = False

    config = get_config(verbose)
    train_data, test_data = acq.extract_data(config["paths"]["train"], config["training"]["dataset_folds"])

    if config["verbose"]["enabled"]:
        acq.show_data(train_data[0], config["verbose"]["sample_timeout"])

    algo = classifier.Classifier()
    algo.train(train_data[0], train_data[1])
    
    estimates = algo.test(test_data[0])
    algo.eval(estimates, test_data[1])

if __name__ == "__main__":
    main()