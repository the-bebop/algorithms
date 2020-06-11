# digit-recognizer
A simple playground to establish a dev-environment. The algorithm's task is to recognize digits in segmented image data.

## Additional files
MNIST was used as dataset and need to be downloaded manually to run the program, see https://www.kaggle.com/c/digit-recognizer/ for further information

## Results

For the outcome of an unoptimized testrun see samples and KPIs in `/results/evaluation`

### Algorithm

This testrun simply used one feature per sample (see `classifier.extract_features()`). It is the correlation of the input data with the avg. appearance of each digit.

Below is the calculated mean appearance of these:

![Averaged appearance of 0](./results/comparables/0.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/1.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/2.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/3.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/4.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/5.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/6.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/7.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/8.png){ width=400%, height=400%}
![Averaged appearance of 0](./results/comparables/9.png){ width=400%, height=400%}

Note the individual amount of samples for this database:
- class 0:  3069
- class 1:  3502
- class 2:  3172
- class 3:  3243
- class 4:  3087
- class 5:  2855
- class 6:  3095
- class 7:  3290
- class 8:  3042
- class 9:  3145


## Known limits

As this is a personal project a few things that I have in mind are not implemented. Since the code is public I want to point out that

- print outs should be replaced with a propper *logging*
- the individual algorithm parts were used with default parameters.
- no k-fold cross validation, as a fair definition for the convolution feature needs to be defined first.
- missing tests, comments and review
