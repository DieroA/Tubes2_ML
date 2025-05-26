import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from Datasets.cifar10 import x_train, y_train, x_val, y_val, x_test, y_test
from Models.CNN.CNN import CNN
from Models.CNN.configs import (
    conv_layers_variations,
    filter_variations,
    kernel_size_variations,
    pooling_variations
)

def train_and_evaluate_all(variation_dict, variation_name):
    print(f"\n=== {variation_name} Variations ===\n")
    for name, config in variation_dict.items():
        print(f"\n--- Training: {name} ---")
        model = CNN.from_config(config)

        model.build()
        model.train(x_train, y_train, x_val, y_val)
        model.evaluate(x_test, y_test)
        model.save()

        print(f"{name} | Macro F1 Score: {model.f1_score:.4f}")

if __name__ == "__main__":
    train_and_evaluate_all(conv_layers_variations, "Conv Layer Count")
    train_and_evaluate_all(filter_variations, "Filter Size")
    train_and_evaluate_all(kernel_size_variations, "Kernel Size")
    train_and_evaluate_all(pooling_variations, "Pooling Type")