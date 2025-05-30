from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
from Models.LSTM.LSTM_forward import LSTMScratch
import pathlib
import numpy as np
import os

class LSTMModel:
    def __init__(self, vocab_size: int, embedding_dim: int, units: list, num_classes: int, sequence_length: int, bidirectional: bool|list, weights_dir: str = "src/Models/LSTM/training/weights"):
        """ Initialize the LSTM model with the specified parameters.
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding layer.
            units (list): List of integers representing the number of units in each LSTM layer.
            num_classes (int): Number of output classes.
            sequence_length (int): Length of the input sequences.
            weights_dir (str): Directory to save or load model weights.
            bidirectional (bool or list): If True, all layers are bidirectional; if a list, specifies which layers are bidirectional.
        Raises:
            ValueError: If the length of `bidirectional` does not match the number of LSTM layers.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.weights_dir = weights_dir
        self.weights_dir = self.weights_dir if isinstance(self.weights_dir, str) else str(self.weights_dir)
        self.weights_dir = pathlib.Path(self.weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(bidirectional, bool):
            self.bidirectional = [bidirectional] * len(self.units)
        else:
            self.bidirectional = list(bidirectional)
        if len(self.bidirectional) != len(self.units):
            raise ValueError(f"Panjang `bidirectional` ({len(self.bidirectional)}) dan `LSTM units` ({len(self.units)}) harus sama.")
        self._dropout_rate = 0.20
        self._learning_rate = 0.005
        self.history = None
        self.f1_score = None
        self.model = None

    @staticmethod
    def from_config(config, vocab_size: int = 10000, embedding_dim: int = 128, sequence_length: int = 100, num_classes: int = 10, weights_dir: str = "src/Models/LSTM/training/weights"):
        """ Create an LSTM model from a configuration dictionary.
        Args:
            config (dict): Configuration dictionary containing model parameters.
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding layer.
            sequence_length (int): Length of the input sequences.
            num_classes (int): Number of output classes.
            weights_dir (str): Directory to save or load model weights.
        Returns:
            LSTM: An instance of the LSTM model.
        """
        return LSTMModel(
            vocab_size = vocab_size,
            embedding_dim = embedding_dim,
            units = config.get("units", [64]),
            num_classes = num_classes,
            sequence_length = sequence_length,
            bidirectional= config.get("bidirectional", False),
            weights_dir = weights_dir
        )
        
    def build(self, dropout_rate: float = 0.20, learning_rate: float = 0.005):
        """ Build the LSTM model with the specified parameters.
        Args:
            dropout_rate (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the optimizer.
        """

        model = Sequential(name="LSTM")
        model.add(
            Embedding(                
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_shape=(self.sequence_length,),
                name="embedding",
            )
        )
        for idx, unit in enumerate (self.units):
            if(idx<len(self.units)-1):
                return_sequences = True
            else:
                return_sequences = False
            lstm = LSTM(
                unit,
                return_sequences = return_sequences,
                name=f"lstm_{idx+1}"
            )

            if self.bidirectional[idx]:
                lstm = Bidirectional(lstm, name=f"bidirectional_{idx+1}")

            model.add(lstm)
            model.add(Dropout(dropout_rate, name=f"dropout_{idx+1}"))
        
        model.add(Dense(self.num_classes, activation="softmax", name="classifier"))
        model.compile(
            optimizer=Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        self.model.summary()


    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, **fit_kwargs):
        """ Train the LSTM model with the provided training and validation data.
        Args:
            X_train (array-like): Training input data.
            y_train (array-like): Training labels.
            X_val (array-like): Validation input data.
            y_val (array-like): Validation labels.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Size of the batches used in training.
        Returns:
            History: Training history object containing loss and accuracy metrics.
        """
        if self.model is None:
            print("Model belum ada")
            print("Membuat model baru...")
            self.build_model(self._dropout_rate, self._learning_rate)
        
        self.history = self.model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = epochs, batch_size = batch_size, verbose=2, **fit_kwargs)
        return self.history

    def evaluate(self, x_test, y_test):
        """ Evaluate the LSTM model on the test data.
        Args:
            x_test (array-like): Test input data.
            y_test (array-like): Test labels.
        Returns:
            float: F1 score of the model on the test data.
        """

        if self.model is None:
            raise ValueError("Model needs to be built before calling evaluate.")
        
        evaluation = self.model.evaluate(x_test, y_test, verbose=2)
        if evaluation is None:
            raise ValueError("Model evaluation failed. Please check the model and data.")
        y_pred = self.model.predict(x_test, verbose=2)
        y_true = y_test.squeeze()
        self.f1_score = f1_score(y_true, np.argmax(y_pred, axis=1), average='weighted')
        if self.f1_score is None:
            raise ValueError("F1 score calculation failed. Please check the predictions and true labels.")
        print(f"Loss: {evaluation[0]:.4f} | Accuracy: {evaluation[1]:.4f} | F1 Score: {self.f1_score:.4f}")
        return self.f1_score

        

    def save(self):
        """ Save the LSTM model to the specified file path.
        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            raise ValueError("Model needs to be built before calling save_model.")

        os.makedirs(self.weights_dir, exist_ok = True)

        name = f"{len(self.units)}layers"
        for idx, unit in enumerate (self.units):
            name += f"_{unit}"
            if(self.bidirectional[idx]):
                name += "_bidir"
            else:
                name += "_unidir"

        filename = name + ".weights.h5"

        filepath = os.path.join(self.weights_dir, filename)
        
        self.model.save_weights(filepath)
        print(f"Saved weights to {filepath}")


    def load(self, filename: str = "lstm_model.h5"):
        """ Load the LSTM model from the specified file path.
        Args:
            filename (str): File name to load the model.
        """
        path = self.weights_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No weights file found at {path}")
        
        if self.model is None:
            self.build(self._dropout_rate, self._learning_rate)
        
        self.model.load_weights(path)
        print(f"Model weights loaded from {path}")
        
    def predict_with_scratch(self, X):
        """
        Lakukan prediksi menggunakan implementasi forward propagation dari scratch.
        
        Args:
            X: Input data (batch_size, sequence_length)
            
        Returns:
            Predicted classes
        """
        if not hasattr(self, 'scratch_predictor'):
            self.scratch_predictor = LSTMScratch(self.model)
        
        return self.scratch_predictor.predict(X)
    
    def evaluate_from_scratch(self, X_test, y_test):
        """Evaluate the LSTM model using our from-scratch implementation"""
        if self.model is None:
            raise ValueError("Model needs to be built before calling evaluate.")

        y_pred = self.predict_with_scratch(X_test)
        # print(f"Output shape: {output.shape}")
        # print(f"Output sample: {output}")
        # y_pred = np.argmax(output, axis=1)
        y_true = y_test.squeeze()
        self.f1_score = f1_score(y_true, y_pred, average='weighted')
        if self.f1_score is None:
            raise ValueError("F1 score calculation failed. Please check the predictions and true labels.")
        print(f"F1 Score: {self.f1_score:.4f}")
        
        return self.f1_score
    
    def compare_predictions(self, X, num_samples=5):
        """
        Bandingkan prediksi dari Keras dan implementasi scratch.
        
        Args:
            X: Input data
            num_samples: Jumlah sampel untuk ditampilkan
        """
        keras_pred = np.argmax(self.model.predict(X[:num_samples]), axis=1)
        scratch_pred = self.predict_with_scratch(X[:num_samples])
        
        print("\nPerbandingan Prediksi:")
        print(f"{'Sample':<10} {'Keras':<10} {'Scratch':<10}")
        print("-" * 30)
        for i in range(num_samples):
            print(f"{i:<10} {keras_pred[i]:<10} {scratch_pred[i]:<10}")
        
        match = np.mean(keras_pred == scratch_pred) * 100
        print(f"\nKesamaan prediksi: {match:.2f}%")
