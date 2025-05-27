import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

from Models.CNN.CNN import CNN
from Models.CNN.training.configs import CONV_LAYER_VARIATIONS

from Datasets.cifar10 import x_test, y_test

def main():
    # Build CNN
    cnn = CNN.from_config(CONV_LAYER_VARIATIONS["3 Layers"])
    cnn.build()

    # Load Weights
    cnn.load("ConvLayerCount/3layers_filters32-64-128_kernels3-3-3_poolmax.weights.h5")

    # Forward prop
    output = cnn.forward_scratch(x_test)
    
    # Evaluate
    cnn.evaluate(output, y_test)
    print(f"f1 score: {cnn.f1_score}")

if __name__ == "__main__":
    main()