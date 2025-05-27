import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tqdm import tqdm
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

# TODO: Forward prop from scratch
class CNN:
    def __init__(self, n_conv_layers = 2, filters_per_layer = [32, 64], kernel_sizes = [3, 3], 
                 pooling_type = "max", n_epoch = 5, weights_dir = "Models/CNN/training/weights"):
        self.n_conv_layers = n_conv_layers
        self.filters_per_layer = filters_per_layer
        self.kernel_sizes = kernel_sizes
        self.pooling_type = pooling_type

        self.n_epoch = n_epoch

        self.weights_dir = weights_dir
        
        self.model = None
        self.history = None
        self.f1_score = None

    @staticmethod
    def from_config(config):
        """"Config format: 
            "Name (Optional)" : {
                "n_conv_layers": x,
                "filters_per_layer": [x, y, ...],
                "kernel_sizes": [x, y, ...]"
                "pooling_type": "max" or "average"
                "n_epochs": x
            }
        """
        return CNN(
            n_conv_layers = config.get("n_conv_layers", 2),
            filters_per_layer = config.get("filters_per_layer", [32, 64]),
            kernel_sizes = config.get("kernel_sizes", [3] * config.get("n_conv_layers", 2)),
            pooling_type = config.get("pooling_type", "max"),
            n_epoch = config.get("n_epochs", 5)
        )

    def build(self):
        self.model = Sequential()
        self.model.add(Input(shape=(32, 32, 3)))

        # Conv2D & Pooling
        for i in range(self.n_conv_layers):
            self.model.add(Conv2D(
                filters = self.filters_per_layer[i],
                kernel_size = self.kernel_sizes[i],
                activation = "relu",
                padding = "same"
            ))

            if self.pooling_type == "max":
                self.model.add(MaxPooling2D(pool_size = (2, 2)))
            else:
                self.model.add(AveragePooling2D(pool_size = (2, 2)))
        
        # Flatten
        self.model.add(Flatten())

        # Dense 1
        self.model.add(Dense(64, activation = "relu"))

        # Dense 2 (Output)
        self.model.add(Dense(10, activation = "softmax"))
        
        self.model.compile(
            optimizer = Adam(),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = ["accuracy"]
        )
    
    def train(self, x_train, y_train, x_val, y_val):
        # Check if model has been built
        if self.model is None:
            raise ValueError("Model needs to be built before calling train_model.")

        # Train
        self.history = self.model.fit(
            x_train, y_train,
            validation_data = (x_val, y_val),
            epochs = self.n_epoch,
            batch_size = 64,
            verbose = 2
        )

    def evaluate(self, y_pred_probs, y_test):
        y_pred = np.argmax(y_pred_probs, axis = 1)
        y_true = y_test.flatten()

        self.f1_score = f1_score(y_true, y_pred, average = "macro")

    def save(self):
        # Check if model has been built
        if self.model is None:
            raise ValueError("Model needs to be built before calling save_model.")

        # Create directory if it doesn't exist yet
        os.makedirs(self.weights_dir, exist_ok = True)

        # Build unique save file name
        filters_str = "-".join(map(str, self.filters_per_layer))
        kernels_str = "-".join(map(str, self.kernel_sizes))

        name = []
        name.append(f"{self.n_conv_layers}layers")
        name.append(f"filters{filters_str}")
        name.append(f"kernels{kernels_str}")
        name.append(f"pool{self.pooling_type}")

        filename = "_".join(name) + ".weights.h5"

        # Save to save_dir
        filepath = os.path.join(self.weights_dir, filename)
        
        self.model.save_weights(filepath)
        print(f"Saved weights to {filepath}")
    
    def load(self, filename):
        # Check if model has been built
        if self.model is None:
            raise ValueError("Model needs to be built before calling load_model.")
        
        load_path = os.path.join(self.weights_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No weights file found at {load_path}")
        
        self.model.load_weights(load_path)
        print(f"Loaded weights from {load_path}")

    """ ------------------------------------- FORWARD PROPAGATION FROM SCRATCH ------------------------------------- """

    @staticmethod
    def _calc_output_size(size, filter_size, padding, stride):
        return ((size - filter_size + 2 * padding) // stride) + 1

    def _conv2d(self, x, W, b, stride = 1, padding = 0):
        # x shape: (H_x, W_x, C_in)
        # W shape: (H_k, W_k, C_in, C_out)
        # b shape: (C_out, 1)
        # W_out = ((W_x - W_k + 2 * padding) / S) + 1
        # output feature map shape: (W_out, W_out, C_out)

        H_x, W_x, _ = x.shape
        H_k, W_k, _, C_out = W.shape

        # Add paddding
        x_padded = np.pad(x, ((padding, padding), (padding, padding), (0, 0)), mode = "constant")

        # Calculate output shape
        W_out = CNN._calc_output_size(W_x, W_k, stride, padding)
        H_out = CNN._calc_output_size(H_x, W_k, stride, padding)
        
        # Calculate feature map
        out = np.zeros((H_out, W_out, C_out))
        for row in range(H_out):
            for col in range(W_out):
                row_start = row * stride
                row_end = row_start + H_k

                col_start = col * stride
                col_end = col_start + W_k

                patch = x_padded[row_start: row_end, col_start: col_end, :]
                for channel in range(C_out):
                    out[row, col, channel] = np.sum(patch * W[:, :, :, channel] + b[channel])

        # ReLU
        out_relu = np.maximum(out, 0)

        return out_relu

    def _pooling(self, x, pool_size = 2, stride = 2, type = "max"):
        H_x, W_x, C_in = x.shape

        if type == "max":
            pooling_func = np.max
        else:
            pooling_func = np.mean

        # Calculate output shape
        W_out = CNN._calc_output_size(W_x, pool_size, 0, stride)
        H_out = CNN._calc_output_size(H_x, pool_size, 0, stride)

        # Calculate output
        out = np.zeros((H_out, W_out, C_in))
        for row in range(H_out):
            for col in range(W_out):
                row_start = row * stride
                row_end = row_start + pool_size

                col_start = col * stride
                col_end = col_start + pool_size

                patch = x[row_start: row_end, col_start: col_end, :]

                out[row, col, :] = pooling_func(patch, axis = (0, 1))
        return out
    
    def _flatten(self, x):
        return x.flatten()

    def _dense(self, x, W, b, activation):
        out = np.dot(x, W) + b
        if activation == "relu":
            return np.maximum(out, 0)
        elif activation == "softmax":
            e_out = np.exp(out - np.max(out))

            return e_out / np.sum(e_out)
        else:
            raise ValueError(f"Invalid activation method `{activation}`.")

    def forward_scratch(self, input):
        if self.model is None:
            raise ValueError("Model must be built and trained before using forward_scratch")
        
        out = []
        for img in tqdm(input, desc = "Forward Propagation", ncols = 80):
            x = img
            layer_idx = 0

            for i in range(self.n_conv_layers):
                # Conv Layer
                conv_layer = self.model.layers[layer_idx]
                
                W, b = conv_layer.get_weights()
                kernel_size = self.kernel_sizes[i]
                padding = kernel_size // 2 # "same" padding

                x = self._conv2d(x, W, b, padding = padding)
                layer_idx += 1

                # Pooling Layer
                x = self._pooling(x, type = self.pooling_type)
                layer_idx += 1

            # Flatten
            x = self._flatten(x)
            layer_idx += 1

            # Dense Layer 1
            dense_layer_1 = self.model.layers[layer_idx]
            W_d1, b_d1 = dense_layer_1.get_weights()
            
            x = self._dense(x, W_d1, b_d1, dense_layer_1.activation.__name__)
            layer_idx += 1

            # Dense Layer 2 (Output)
            output_layer = self.model.layers[layer_idx]
            W_o, b_o = output_layer.get_weights()

            x = self._dense(x, W_o, b_o, output_layer.activation.__name__)

            # Append final result
            out.append(x)
        return np.array(out)