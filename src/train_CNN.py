import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import matplotlib.pyplot as plt
from contextlib import redirect_stdout

from Datasets.cifar10 import x_train, y_train, x_val, y_val, x_test, y_test
from Models.CNN.CNN import CNN
from Models.CNN.training.configs import (
    CONV_LAYER_VARIATIONS, FILTER_VARIATIONS, 
    KERNEL_SIZE_VARIATIONS, POOLING_VARIATIONS
)

SAVE_PATH = "Models/CNN/training"

def plot_loss_curves(history, title, save_path = None):
    """Plot loss curves; saves or shows plot."""

    plt.plot(history.history["loss"], label = "Train Loss")
    plt.plot(history.history["val_loss"], label = "Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def train_and_evaluate_all(variation_dict, variation_name):
    """Train and evaluate CNN models for each config in variation_dict."""

    print(f"\n\n=== {variation_name} Variations ===\n")
    for name, config in variation_dict.items():
        print(f"\n--- Training: {name} ---")

        # Build
        model = CNN.from_config(config)
        model.build()

        # Train & Evaluate
        model.train(x_train, y_train, x_val, y_val)
        y_pred_probs = model.predict(x_test)
        model.evaluate(y_pred_probs, y_test)

        # Save weights
        model.save()

        # Log results
        print(f"{name} | Macro F1 Score: {model.f1_score:.4f}")
        plot_loss_curves(model.history, f"{name} | Training & Validation Loss", save_path = f"{SAVE_PATH}/plots/{name}_loss_curve.png")

def main():
    # Ensures save directories exists
    os.makedirs(SAVE_PATH, exist_ok = True)
    os.makedirs(os.path.join(SAVE_PATH, "plots"), exist_ok = True)
    os.makedirs(os.path.join(SAVE_PATH, "logs"), exist_ok = True)

    # Redirect all prints made to log file
    with open(f"{SAVE_PATH}/logs/training_log.txt", "w") as f:
        with redirect_stdout(f):
            train_and_evaluate_all(CONV_LAYER_VARIATIONS, "Conv Layer Count")
            train_and_evaluate_all(FILTER_VARIATIONS, "Filter Size")
            train_and_evaluate_all(KERNEL_SIZE_VARIATIONS, "Kernel Size")
            train_and_evaluate_all(POOLING_VARIATIONS, "Pooling Type")

if __name__ == "__main__":
    main()
