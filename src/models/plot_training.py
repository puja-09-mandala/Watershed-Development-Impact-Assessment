import numpy as np
import matplotlib.pyplot as plt
import os

print("========================================")
print("CNN-LSTM TRAINING GRAPH")
print("========================================")

history_path = "outputs/models/training_history.npy"

print("Loading:", history_path)

history = np.load(
    history_path,
    allow_pickle=True
).item()

print("History loaded successfully!")
print("Epochs:", len(history["loss"]))

os.makedirs("outputs/graphs", exist_ok=True)


# ==========================================
# ACCURACY GRAPH
# ==========================================

print("Creating accuracy graph...")

plt.figure(figsize=(8, 5))

plt.plot(
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNN-LSTM Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/graphs/cnn_lstm_accuracy.png",
    dpi=300
)

plt.close()

print("Accuracy graph saved.")


# ==========================================
# LOSS GRAPH
# ==========================================

print("Creating loss graph...")

plt.figure(figsize=(8, 5))

plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNN-LSTM Training and Validation Loss"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/graphs/cnn_lstm_loss.png",
    dpi=300
)

plt.close()

print("Loss graph saved.")


# ==========================================
# COMPLETE
print("\n========================================")
print("GRAPH CREATION COMPLETED")
print("========================================")

print("Accuracy graph:")
print("outputs/graphs/cnn_lstm_accuracy.png")

print("Loss graph:")
print("outputs/graphs/cnn_lstm_loss.png")