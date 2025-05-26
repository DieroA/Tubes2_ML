from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

# Load
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)

# Normalize
def normalize_x(x):
    return x.astype("float32") / 255.0

x_train = normalize_x(x_train)
x_val = normalize_x(x_val)
x_test = normalize_x(x_test)

if __name__ == "__main__":
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    print(f"Train shape: {x_train.shape}\nValidation shape: {x_val.shape}\nTest shape: {x_test.shape}")