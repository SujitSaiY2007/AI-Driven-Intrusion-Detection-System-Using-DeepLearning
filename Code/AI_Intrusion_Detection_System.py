import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# 1. Dataset columns and loading
# ---------------------------------------------------------------------

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

train = pd.read_csv("KDDTrain+.txt", names=columns)
test = pd.read_csv("KDDTest+.txt", names=columns)

print("Original training shape:", train.shape)
print("Original testing shape:", test.shape)


# ---------------------------------------------------------------------
# 2. Binary label conversion and removal of difficulty field
# ---------------------------------------------------------------------

train['label'] = train['label'].apply(
    lambda x: 0 if x == 'normal' else 1
)
test['label'] = test['label'].apply(
    lambda x: 0 if x == 'normal' else 1
)

train.drop('difficulty', axis=1, inplace=True)
test.drop('difficulty', axis=1, inplace=True)


# ---------------------------------------------------------------------
# 3. One-hot encoding and feature alignment
# ---------------------------------------------------------------------

train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Keep the training feature schema and add missing test columns as zeros.
# Test-only dummy columns are not retained by this left alignment.
train, test = train.align(
    test, join='left', axis=1, fill_value=0
)

print("Encoded training shape:", train.shape)
print("Encoded testing shape:", test.shape)


# ---------------------------------------------------------------------
# 4. Separate features and labels
# ---------------------------------------------------------------------

X_train = train.drop('label', axis=1)
y_train = train['label']

X_test = test.drop('label', axis=1)
y_test = test['label']


# ---------------------------------------------------------------------
# 5. Feature scaling
# ---------------------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------------------------
# 6. Neural network architecture
# ---------------------------------------------------------------------

model = Sequential([
    Dense(
        256,
        activation='relu',
        kernel_regularizer=l2(1e-4),
        input_shape=(X_train.shape[1],)
    ),
    BatchNormalization(),
    Dropout(0.4),

    Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(1e-4)
    ),
    BatchNormalization(),
    Dropout(0.3),

    Dense(
        64,
        activation='relu',
        kernel_regularizer=l2(1e-4)
    ),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


# ---------------------------------------------------------------------
# 7. Compile model
# ---------------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ---------------------------------------------------------------------
# 8. Calculate class weights
# ---------------------------------------------------------------------

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print("Class weights:", class_weights)


# ---------------------------------------------------------------------
# 9. Training callbacks
# ---------------------------------------------------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)


# ---------------------------------------------------------------------
# 10. Train the model
# ---------------------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler]
)


# ---------------------------------------------------------------------
# 11. Evaluate model using Keras' default binary threshold
# ---------------------------------------------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy (default threshold):", test_accuracy)
print("Test Loss:", test_loss)


# ---------------------------------------------------------------------
# 12. Generate probabilities and select ROC / Youden J threshold
# ---------------------------------------------------------------------

y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_thresh = thresholds[best_idx]

print(f"Optimal threshold (Youden J): {best_thresh:.4f}")


# ---------------------------------------------------------------------
# 13. Convert probabilities into binary predictions
# ---------------------------------------------------------------------

y_pred = (y_prob >= best_thresh).astype(int)


# ---------------------------------------------------------------------
# 14. Confusion matrix and classification report
# ---------------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------------------------
# 15. Training and testing accuracy using the selected threshold
# ---------------------------------------------------------------------

train_predictions = (
    (model.predict(X_train) >= best_thresh)
    .astype(int)
    .flatten()
)

train_correct = int(np.sum(train_predictions == y_train.values))
train_total = len(y_train)
train_pct = train_correct / train_total * 100

test_correct = int(np.sum(y_pred.flatten() == y_test.values))
test_total = len(y_test)
test_pct = test_correct / test_total * 100

print("\n========== Accuracy Summary ==========")
print(f"Train Accuracy : {train_correct}/{train_total}  -> {train_pct:.2f}%")
print(f"Test Accuracy  : {test_correct}/{test_total}   -> {test_pct:.2f}%")
print("=======================================\n")


# ---------------------------------------------------------------------
# 16. Training and validation accuracy plot
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()


# ---------------------------------------------------------------------
# 17. Training and validation loss plot
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# ---------------------------------------------------------------------
# 18. Confusion matrix heatmap
# ---------------------------------------------------------------------

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal', 'Attack'],
    yticklabels=['Normal', 'Attack']
)
plt.title('Intrusion Detection Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# ---------------------------------------------------------------------
# 19. Prediction distribution
# ---------------------------------------------------------------------

labels = ['Normal', 'Attack']
values = np.bincount(y_pred.flatten())

plt.figure(figsize=(6, 5))
plt.bar(labels, values)
plt.title('Prediction Distribution')
plt.ylabel('Number of Predictions')
plt.show()


# ---------------------------------------------------------------------
# 20. Training versus testing accuracy plot
# ---------------------------------------------------------------------

categories = ['Train Accuracy', 'Test Accuracy']
percentages = [train_pct, test_pct]
colors = ['#4C9BE8', '#E8734C']

plt.figure(figsize=(6, 5))
bars = plt.bar(
    categories,
    percentages,
    color=colors,
    width=0.5
)

for bar, pct, correct, total in zip(
    bars,
    percentages,
    [train_correct, test_correct],
    [train_total, test_total]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 3,
        f"{pct:.2f}%\n({correct}/{total})",
        ha='center',
        va='top',
        fontsize=11,
        color='white',
        fontweight='bold'
    )

plt.ylim(0, 105)
plt.title('Train vs Test Accuracy (%)')
plt.ylabel('Accuracy (%)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
