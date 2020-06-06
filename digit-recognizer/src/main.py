import os
import acquisition as acq
import classifier

def get_config(with_verbose) -> dict:
    
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

    return config


def main():
    verbose = True

    config = get_config(verbose)
    y, X = acq.extract_data(config["paths"]["train"])

    if config["verbose"]["enabled"]:
        acq.show_data(X, config["verbose"]["sample_timeout"])

    algo = classifier.Classifier()
    algo.train(X, y)
    
    y, X = acq.extract_data(config["paths"]["train"])
    estimates = algo.test(X)
    algo.eval(estimates, y)

if __name__ == "__main__":
    main()