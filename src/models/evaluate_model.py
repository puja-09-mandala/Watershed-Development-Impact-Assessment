import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)
# ============================================================
# 1. PATHS
# ============================================================
DATA_DIR = "data/processed/dataset"
MODEL_PATH = "outputs/models/best_cnn_lstm.keras"
OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("========================================")
print("LOADING TEST DATA")
print("========================================")

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================
print("\n========================================")
print("LOADING CNN-LSTM MODEL")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)
print("Model loaded successfully.")
# ============================================================
# 4. MODEL EVALUATION
# ============================================================
print("\n========================================")
print("MODEL EVALUATION")
print("========================================")
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=4,
    verbose=1
)
print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
# ============================================================
# 5. PREDICTION
# ============================================================
print("\n========================================")
print("GENERATING PREDICTIONS")
print("========================================")
predictions = model.predict(
    X_test,
    batch_size=4,
    verbose=1
)
print("Prediction shape:", predictions.shape)
# Convert probability output to class labels
y_pred = np.argmax(
    predictions,
    axis=-1
)
print("Predicted labels shape:", y_pred.shape)
# ============================================================
# 6. FLATTEN LABELS
# ===========================================================
y_true_flat = y_test.reshape(-1)
y_pred_flat = y_pred.reshape(-1)
# ============================================================
# 7. CLASS NAMES
# ============================================================

class_names = [
    "Background",
    "Water",
    "Vegetation",
    "Built-up",
    "Barren"
]
class_ids = [0, 1, 2, 3, 4]
# ============================================================
# 8. PRECISION
# ============================================================
precision = precision_score(
    y_true_flat,
    y_pred_flat,
    labels=class_ids,
    average="weighted",
    zero_division=0
)
# ============================================================
# 9. RECALL
# ============================================================
recall = recall_score(
    y_true_flat,
    y_pred_flat,
    labels=class_ids,
    average="weighted",
    zero_division=0
)
# ============================================================
# 10. F1 SCORE
# ============================================================
f1 = f1_score(
    y_true_flat,
    y_pred_flat,
    labels=class_ids,
    average="weighted",
    zero_division=0
)
iou = jaccard_score(
    y_true_flat,
    y_pred_flat,
    labels=class_ids,
    average="weighted",
    zero_division=0
)


# ============================================================
# 12. PRINT METRICS
# ============================================================

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"IoU       : {iou:.4f}"
)
print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_true_flat,
    y_pred_flat,
    labels=class_ids,
    target_names=class_names,
    zero_division=0
)

print(report)


# Save report

with open(
    f"{OUTPUT_DIR}/classification_report.txt",
    "w"
) as file:

    file.write(
        "CNN-LSTM MODEL EVALUATION\n"
    )

    file.write(
        "========================================\n\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n"
    )

    file.write(
        f"IoU: {iou:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "========================================\n"
    )

    file.write(report)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CREATING CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(
    y_true_flat,
    y_pred_flat,
    labels=class_ids
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 15. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(cm)

plt.title(
    "CNN-LSTM Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=30
)

plt.yticks(
    range(len(class_names)),
    class_names
)


# Add numbers inside cells

for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

np.save(
    f"{OUTPUT_DIR}/y_pred.npy",
    y_pred
)


# ============================================================
# 17. SAVE SAMPLE PREDICTION MAP
# ============================================================

sample_index = 0

actual_map = y_test[
    sample_index
]

predicted_map = y_pred[
    sample_index
]


plt.figure(
    figsize=(12, 5)
)

plt.subplot(1, 2, 1)

plt.imshow(
    actual_map,
    vmin=0,
    vmax=4
)

plt.title(
    "Actual LULC 2024"
)

plt.xlabel("Pixel")
plt.ylabel("Pixel")

plt.colorbar()


plt.subplot(1, 2, 2)

plt.imshow(
    predicted_map,
    vmin=0,
    vmax=4
)

plt.title(
    "CNN-LSTM Predicted LULC"
)

plt.xlabel("Pixel")
plt.ylabel("Pixel")

plt.colorbar()


plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/actual_vs_predicted.png",
    dpi=300
)

plt.close()
# ============================================================
# 18. FINAL
# ============================================================
print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")

print(
    "Results saved in:"
)

print(
    OUTPUT_DIR
)

print("\nGenerated files:")

print("1. classification_report.txt")
print("2. confusion_matrix.png")
print("3. actual_vs_predicted.png")
print("4. y_pred.npy")