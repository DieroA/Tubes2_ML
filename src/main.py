import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

from Models.CNN.CNN import CNN
from Models.CNN.training.configs import CONV_LAYER_VARIATIONS

from Datasets.cifar10 import x_test, y_test

def compare_scratch_keras_cnn():
    # Build CNN
    cnn = CNN.from_config(CONV_LAYER_VARIATIONS["3 Layers"])
    cnn.build()

    # Load Weights
    cnn.load("ConvLayerCount/3layers_filters32-64-128_kernels3-3-3_poolmax.weights.h5")

    # Forward prop
    output_scratch = cnn.forward_scratch(x_test)
    output_keras = cnn.model.predict(x_test)
    
    # Evaluate
    cnn.evaluate(output_scratch, y_test)
    score_scratch = cnn.f1_score

    cnn.evaluate(output_keras, y_test)
    score_keras = cnn.f1_score

    # Evaluate 
    print(f"From Scratch | F1-Score: {score_scratch}")
    print(f"Keras | F1-Score: {score_keras}")

def main():
    compare_scratch_keras_cnn()
    
if __name__ == "__main__":
    main()