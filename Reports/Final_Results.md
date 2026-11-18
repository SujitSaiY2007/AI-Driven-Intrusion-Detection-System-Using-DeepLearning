# Final Results and Evaluation Notes

This file records the metrics reported in the submitted project report, `MDP_Report_IDS.docx`, and explains how they should be interpreted.

## Dataset

| Property | Training | Test |
|---|---:|---:|
| Samples | 125,973 | 22,544 |
| Normal samples | 67,343 | 9,711 |
| Attack samples | 58,630 | 12,833 |
| Encoded columns, including target | 123 | 123 |
| Input features | 122 | 122 |

The raw NSL-KDD files contain 43 columns: 41 traffic attributes, the class label, and the difficulty field. The difficulty field is removed. One-hot encoding expands the categorical attributes, resulting in 122 model input features plus the target label.

## Model configuration

- Fully connected ANN/DNN with hidden layer sizes 256, 128, 64, and 32.
- ReLU activation in hidden layers and sigmoid output for binary classification.
- Batch Normalization after the first three Dense layers.
- Dropout rates 0.4, 0.3, and 0.3.
- L2 kernel regularization of 1e-4 on the first three Dense layers.
- Adam optimizer, initial learning rate 0.001.
- Binary Crossentropy loss.
- Maximum 50 epochs, batch size 256, validation split 20%.
- EarlyStopping (patience 7, restore best weights) and ReduceLROnPlateau (factor 0.5, patience 3, minimum learning rate 1e-6).
- Balanced class weights: Normal 0.9353; Attack 1.0743.

## Reported training history

| Epoch | Training accuracy | Validation accuracy | Training loss | Validation loss |
|---:|---:|---:|---:|---:|
| 1 | 97.09% | 99.11% | 0.1242 | 0.0607 |
| 5 | 99.03% | 99.25% | 0.0494 | 0.0416 |
| 10 | 99.14% | 99.25% | 0.0373 | 0.0314 |
| 20 | 99.29% | 99.53% | 0.0301 | 0.0241 |
| 30 | 99.40% | 99.62% | 0.0280 | 0.0222 |
| 40 | 99.56% | 99.65% | 0.0211 | 0.0178 |
| 50 | 99.65% | 99.71% | 0.0168 | 0.0149 |

The report's separate accuracy summary gives 97.21% training accuracy. This differs from the epoch-50 Keras training-history accuracy of 99.65%. The two figures are produced in different evaluation contexts and should not be described as the same metric.

## Test results reported

The report uses a Youden J threshold of 0.0031.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal (0) | 0.77 | 0.95 | 0.85 | 9,711 |
| Attack (1) | 0.95 | 0.79 | 0.86 | 12,833 |
| Macro average | 0.86 | 0.87 | 0.86 | 22,544 |
| Weighted average | 0.88 | 0.86 | 0.86 | 22,544 |

### Confusion matrix

|  | Predicted Normal | Predicted Attack |
|---|---:|---:|
| Actual Normal | 9,212 (TN) | 499 (FP) |
| Actual Attack | 2,699 (FN) | 10,134 (TP) |

Correct predictions: 9,212 + 10,134 = 19,346.

Test accuracy: 19,346 / 22,544 = 85.81%.

## How to interpret the results

- Normal recall of approximately 95% means most legitimate records in the test set were correctly classified as normal.
- Attack precision of approximately 95% means most records predicted as attacks were attacks in this test set.
- Attack recall of approximately 79% means the model detected about four out of five actual attack records; approximately one out of five attack records was missed.
- The 2,699 false negatives are particularly important in an intrusion-detection context because they represent attacks classified as normal.
- The 499 false positives represent normal records incorrectly flagged as attacks.

## Evaluation limitation: threshold-selection leakage

The implementation obtains probabilities for the test set, calculates the ROC curve using `y_test`, selects the threshold maximizing Youden's J statistic, and then evaluates predictions on that same test set.

This means the test labels have influenced the selected threshold. Consequently, the 85.81% test accuracy and associated threshold-based classification metrics are the observed results from this experiment, but they should not be presented as a fully independent final test estimate.

Recommended correction for a future rigorous experiment:

1. Split the original training data into training and validation subsets.
2. Fit the model on the training subset.
3. Select the threshold using validation labels and validation probabilities only.
4. Freeze that threshold.
5. Evaluate once on KDDTest+ without using its labels to make any model or threshold decisions.

## Scope

This is an offline, dataset-based binary classifier. It does not perform live packet capture, real-time network monitoring, automated blocking, or incident response. NSL-KDD benchmark performance alone does not establish production-network effectiveness.
