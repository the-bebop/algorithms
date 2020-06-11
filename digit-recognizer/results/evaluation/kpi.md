# Result
This is the outcome of an unoptimized test run.

## Run the algorithm

To reproduce the algorithm simply run main and set the *random_seed* in *get_config()* to *312*.

## KPI

See below the *precision* and *recall* metric for each MNIST digit/class.

|        |Precision             | Recall                |
|--------|----------------------|-----------------------|
|Class 0 |0.9567262464722484    |0.9356025758969642     |
|Class 1 |0.9788494077834179    |0.9429502852485737     |
|Class 2 |0.8716417910447761    |0.8579823702252694     |
|Class 3 |0.7897111913357401    |0.875                  |
|Class 4 |0.86497461928934      |0.8711656441717791     |
|Class 5 |0.8691489361702127    |0.8097125867195243     |
|Class 6 |0.9222648752399232    |0.9161105815061964     |
|Class 7 |0.9234923492349235    |0.926829268292683      |
|Class 8 |0.8334965719882468    |0.8135755258126195     |
|Class 9 |0.8015340364333653    |0.8565573770491803     |

## Samples

Below are randomly chosen samples estimated and compared with its ground truth for a better insight

| Estimate  | Ground Truth  | Visu                                                              |
|-----------|---------------|-------------------------------------------------------------------|
| 0         | 0             | ![estimate '0' ground truth '0'](./estimated_as_0_gt-label_0.png) |
| 1         | 1             | ![estimate '1' ground truth '1'](./estimated_as_1_gt-label_1.png) |
| 3         | 3             | ![estimate '3' ground truth '3'](./estimated_as_3_gt-label_3.png) |
| 4         | 4             | ![estimate '4' ground truth '4'](./estimated_as_4_gt-label_4.png) |
| 4         | 9             | ![estimate '4' ground truth '9'](./estimated_as_4_gt-label_9.png) |
| 5         | 5             | ![estimate '5' ground truth '5'](./estimated_as_5_gt-label_5.png) |
| 5         | 8             | ![estimate '5' ground truth '8'](./estimated_as_5_gt-label_8.png) |
| 7         | 7             | ![estimate '7' ground truth '7'](./estimated_as_7_gt-label_7.png) |









