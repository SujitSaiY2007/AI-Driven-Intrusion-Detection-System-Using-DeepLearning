# Final Results and Evaluation Notes

**Source:** Saved execution outputs in `AI-Driven-IDS-Execution-Colab-File(1).ipynb`. The figures below are transcribed from that notebook, not from a separate project report.

## Dataset and preprocessing

| Item | Training set | Test set |
|---|---:|---:|
| Original records | 125,973 | 22,544 |
| Original columns | 43 | 43 |
| Columns after one-hot encoding, including label | 123 | 123 |
| Model input features | 122 | 122 |

The notebook removes the `difficulty` column, maps `normal` to 0 and all other labels to 1, applies one-hot encoding, aligns test columns to the training columns, and standardizes the features.

## Model and training configuration

- ANN hidden layers: 256, 128, 64, and 32 neurons.
- ReLU hidden activations; sigmoid output.
- Batch Normalization after the first three Dense layers.
- Dropout: 0.4, 0.3, and 0.3.
- L2 kernel regularization: 1e-4 on the first three Dense layers.
- Adam optimizer; initial learning rate 0.001.
- Binary Crossentropy loss.
- Maximum 50 epochs; batch size 256; validation split 20%.
- EarlyStopping: monitor `val_loss`, patience 7, restore best weights.
- ReduceLROnPlateau: monitor `val_loss`, factor 0.5, patience 3, minimum learning rate 1e-6.
- Class weights: Normal (0) = 0.9353088; Attack (1) = 1.0743050.

## Training history

The notebook completed all 50 epochs. Selected logged values:

| Epoch | Training accuracy | Validation accuracy | Training loss | Validation loss | Learning rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 97.06% | 99.06% | 0.1211 | 0.0619 | 0.001 |
| 10 | 99.15% | 99.37% | 0.0361 | 0.0303 | 0.001 |
| 20 | 99.43% | 99.60% | 0.0260 | 0.0223 | 0.0005 |
| 30 | 99.53% | 99.63% | 0.0215 | 0.0186 | 0.0005 |
| 40 | 99.65% | 99.73% | 0.0177 | 0.0150 | 0.00025 |
| 50 | 99.66% | 99.73% | 0.0157 | 0.0138 | 0.00025 |

Final epoch values: training accuracy 99.66%, validation accuracy 99.73%, training loss 0.0157, validation loss 0.0138.

## Test evaluation

### Keras evaluation at the default threshold

The call to `model.evaluate(X_test, y_test)` reports:

| Metric | Value |
|---|---:|
| Test accuracy | 81.32% |
| Test loss | 1.5212 |

This uses the default binary accuracy decision threshold (0.5).

### Evaluation using the ROC-selected threshold

The notebook calculates the ROC curve and selects the threshold maximizing Youden's J statistic:

```
Youden's J = True Positive Rate - False Positive Rate
```

| Metric | Value |
|---|---:|
| Selected threshold | 0.0031 |
| Correct test predictions | 19,367 / 22,544 |
| Threshold-based test accuracy | 85.91% |
| Training accuracy at the same threshold | 97.20% (122,443 / 125,973) |

The 81.32% Keras test accuracy and 85.91% threshold-based test accuracy are different because they use different decision thresholds.

## Classification report at threshold 0.0031

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal (0) | 0.78 | 0.93 | 0.85 | 9,711 |
| Attack (1) | 0.94 | 0.81 | 0.87 | 12,833 |
| Macro average | 0.86 | 0.87 | 0.86 | 22,544 |
| Weighted average | 0.87 | 0.86 | 0.86 | 22,544 |

## Confusion matrix

Rows are actual classes; columns are predicted classes.

|  | Predicted Normal | Predicted Attack |
|---|---:|---:|
| Actual Normal | 9,008 (TN) | 703 (FP) |
| Actual Attack | 2,474 (FN) | 10,359 (TP) |

Correct predictions: 9,008 + 10,359 = 19,367.

Test accuracy: 19,367 / 22,544 = 85.91%.

### Interpretation

- **Normal recall (0.93):** approximately 93% of actual normal records were classified correctly.
- **Attack precision (0.94):** approximately 94% of records predicted as attacks were actual attacks.
- **Attack recall (0.81):** approximately 81% of actual attack records were detected.
- **False negatives (2,474):** attack records incorrectly classified as normal.
- **False positives (703):** normal records incorrectly classified as attacks.

## Evaluation limitation: threshold-selection leakage

The notebook computes `roc_curve(y_test, y_prob)`, chooses the Youden J threshold using the test labels, and then evaluates predictions on the same test set. The test labels therefore influence threshold selection.

The reported 85.91% threshold-based accuracy and the associated classification metrics are the actual notebook outputs, but they are **not an independent, untouched-test estimate**. For a stronger evaluation, select the threshold using validation predictions and labels, freeze it, and evaluate once on the test set.

## Scope

This is an offline, dataset-based binary classifier. It does not capture live packets, monitor a live network, block traffic, or perform incident response. Performance on NSL-KDD alone does not establish effectiveness on modern production networks.
