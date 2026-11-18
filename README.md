# AI-Driven Network Intrusion Detection System Using Deep Learning

## Description

This project implements a binary network intrusion detection model using an Artificial Neural Network (ANN) trained on the **NSL-KDD** dataset. It classifies network connection records into two categories:

- **Normal (0):** legitimate network activity
- **Attack (1):** malicious or intrusive network activity

The workflow covers dataset loading, preprocessing, categorical feature encoding, feature alignment, standardization, class-imbalance handling, neural-network training, threshold-based classification, and performance evaluation.

> **Scope:** This is an offline, dataset-based IDS prototype. It does not capture live network packets or automatically block network traffic.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main implementation language |
| Pandas | Dataset loading and tabular preprocessing |
| NumPy | Numerical operations and array handling |
| TensorFlow / Keras | Neural-network construction and training |
| Scikit-learn | Feature scaling, class weights, ROC analysis, and evaluation metrics |
| Matplotlib | Training curves and result visualizations |
| Seaborn | Confusion-matrix heatmap |

## Architecture Diagram

The diagram below is embedded directly in this README using Mermaid, so GitHub can render it without a separate image file.

```mermaid
flowchart TD
    A["NSL-KDD Dataset<br/>KDDTrain+ and KDDTest+"] --> B["Load records<br/>Assign column names"]
    B --> C["Label preparation<br/>Normal = 0, Attack = 1"]
    C --> D["Remove difficulty column"]
    D --> E["One-hot encode categorical features"]
    E --> F["Align train and test feature columns"]
    F --> G["Separate features (X) and labels (y)"]
    G --> H["StandardScaler<br/>Fit on training data; transform both sets"]
    H --> I["Class-weight calculation<br/>Balanced class weights"]
    I --> J["ANN training<br/>Dense: 256 → 128 → 64 → 32"]
    J --> K["Regularization and training controls<br/>Batch Normalization, Dropout, L2,<br/>EarlyStopping, ReduceLROnPlateau"]
    K --> L["Sigmoid output<br/>Attack probability"]
    L --> M["ROC analysis and threshold selection"]
    M --> N["Binary classification<br/>Normal (0) / Attack (1)"]
    N --> O["Evaluation<br/>Confusion Matrix, Precision, Recall,<br/>F1-score, Accuracy"]
    O --> P["Visualizations<br/>Learning curves, heatmap,<br/>prediction distribution, accuracy comparison"]

    style A fill:#e8f1ff,stroke:#3973ac,color:#172b4d
    style J fill:#e9f7ef,stroke:#39845a,color:#173b28
    style N fill:#fff4df,stroke:#bd8425,color:#513b12
    style O fill:#f3eaff,stroke:#8656ad,color:#352044
```

### Neural Network Configuration

| Stage | Configuration |
|---|---|
| Input | Number of encoded and scaled features |
| Hidden layer 1 | 256 neurons, ReLU, L2 regularization, Batch Normalization, Dropout (0.4) |
| Hidden layer 2 | 128 neurons, ReLU, L2 regularization, Batch Normalization, Dropout (0.3) |
| Hidden layer 3 | 64 neurons, ReLU, L2 regularization, Batch Normalization, Dropout (0.3) |
| Hidden layer 4 | 32 neurons, ReLU |
| Output | 1 neuron, Sigmoid |
| Loss | Binary Crossentropy |
| Optimizer | Adam (initial learning rate: 0.001) |
| Training limit | Up to 50 epochs; EarlyStopping may stop training earlier |
| Batch size | 256 |
| Validation | 20% of the supplied training data |

## Results

The notebook evaluates the trained classifier using:

- Test loss and default-threshold test accuracy from Keras evaluation.
- Confusion matrix.
- Per-class precision, recall, F1-score, and support.
- Training and validation accuracy/loss curves.
- Prediction distribution.
- Training-versus-test accuracy calculated using the selected classification threshold.

**Insert the exact values from the final executed notebook here before publishing numerical performance claims.** The source code alone does not establish the final metric values, so this README intentionally does not invent accuracy, precision, recall, or F1-score figures.

**Evaluation note:** In the supplied implementation, the ROC threshold is selected using `y_test` and then evaluated on that same test set. This uses test labels during threshold selection, so the resulting threshold-based test metrics are not an independent final estimate. For a rigorous final evaluation, select and freeze the threshold using validation data, then evaluate once on the untouched test set.

## Limitations and Improvements

### Current limitations

1. **Offline evaluation:** The implementation classifies records from a prepared dataset; it is not connected to live packet capture or real-time traffic monitoring.
2. **Binary classification:** All non-normal labels are mapped to the single `Attack` class. The model does not identify individual attack families.
3. **Dataset scope:** Results on NSL-KDD do not by themselves establish performance on modern production networks or unseen traffic environments.
4. **Threshold-selection leakage:** The current code selects the ROC threshold using test labels. Threshold selection should be moved to validation data.
5. **Preprocessing deployment:** The fitted encoder/feature schema and scaler need to be saved and reused consistently when predicting on new records.

### Possible improvements

- Select the classification threshold using validation data and preserve the test set for final evaluation.
- Report ROC-AUC and, where useful for class imbalance, Precision-Recall AUC.
- Evaluate per-attack-category performance using a multiclass approach.
- Test generalization on additional or more recent network-traffic datasets.
- Save the trained model and fitted preprocessing objects for repeatable inference.
- Add live traffic ingestion and monitoring as a separate future development stage.
- Compare the ANN against suitable baseline models.

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/SujitSaiY2007/AI-Driven-Intrusion-Detection-System-Using-DeepLearning.git
cd AI-Driven-Intrusion-Detection-System-Using-DeepLearning
```

### 2. Install dependencies

Use Python 3 and install the libraries used by the notebook:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn jupyter
```

For repeatable experiments, record and pin the exact package versions used in the environment in a `requirements.txt` file.

### 3. Obtain the dataset

Download the NSL-KDD dataset from its legitimate distribution source. Place the following files in the working directory expected by the notebook:

- `KDDTrain+.txt`
- `KDDTest+.txt`

The dataset is not included in this repository. Review the dataset source's terms before redistribution.

### 4. Run the notebook

Open the notebook in Jupyter or Google Colab and execute the cells in order. Confirm that the dataset paths match the files' locations.

### 5. Reproducibility considerations

- The supplied code does not explicitly set Python, NumPy, and TensorFlow random seeds. Neural-network results may therefore vary between runs.
- Record the Python and library versions, dataset source, and execution environment alongside reported metrics.
- Keep the test set separate from threshold selection for an unbiased final evaluation.

## Repository Structure

```text
.
├── Code/
│   └── AI_Intrusion_Detection_System.ipynb
├── Architecture/
│   └── System_Architecture.md
├── README.md
└── LICENSE
```

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

The code license does not automatically apply to the NSL-KDD dataset. Refer to the dataset provider's terms for dataset usage and redistribution.
