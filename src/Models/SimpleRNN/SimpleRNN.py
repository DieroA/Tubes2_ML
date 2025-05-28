import os, pathlib
import numpy as np
import tensorflow as tf
from keras import Sequential, layers, optimizers
from sklearn.metrics import f1_score


class SimpleRNNModel:
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_units, num_classes: int, sequence_length: int, bidirectional=False, w_dir: str = "src/Models/SimpleRNN/training/weights"):
        self.vocab_size        = vocab_size
        self.embedding_dim     = embedding_dim

        # Kalau integer, jadiin list
        if isinstance(rnn_units, int):
            self.rnn_units = [rnn_units]
        # Kalau list, tetep jadi list
        else:
            self.rnn_units = list(rnn_units)
        
        # Kalau boolean, jadiin list
        if isinstance(bidirectional, bool):
            self.bidirectional = [bidirectional] * len(self.rnn_units)
        # Kalau list, tetep jadi list
        else:
            self.bidirectional = list(bidirectional)

        if len(self.bidirectional) != len(self.rnn_units):
            raise ValueError(f"Panjang `bidirectional` ({len(self.bidirectional)}) dan `rnn_units` ({len(self.rnn_units)}) harus sama.")
            
        self.num_classes       = num_classes
        self.sequence_length   = sequence_length

        self.model     = None
        self.history   = None
        self.f1_macro  = None

        self.w_dir = pathlib.Path(w_dir)
        self.w_dir.mkdir(parents=True, exist_ok=True)

        self._dropout_rate  = 0.20
        self._learning_rate = 0.005

    def build_model(self, dropout_rate : float = 0.20, learning_rate: float = 0.005):
        self._dropout_rate  = dropout_rate
        self._learning_rate = learning_rate

        m = Sequential(name="flex_rnn")
        
        m.add(
            layers.Embedding
            (
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_shape=(self.sequence_length,),
                name="embedding",
            )
        )

        for idx in range(len(self.rnn_units)):
            units = self.rnn_units[idx]
            b = self.bidirectional[idx]

            if idx < len(self.rnn_units) - 1:
                return_sequences = True
            else:
                return_sequences = False

            rnn = layers.SimpleRNN(
                units,
                return_sequences=return_sequences,
                name=f"rnn_{idx+1}",
            )

            # Kalau bidirectional, bungkus pake Bidirectional layer
            if b:
                rnn = layers.Bidirectional(rnn, name=f"bidir_{idx+1}")

            m.add(rnn)

        m.add(layers.Dropout(dropout_rate, name="dropout"))
        m.add(layers.Dense(self.num_classes, activation="softmax", name="classifier"))

        m.compile(
            optimizer=optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = m
        self.model.summary()

    def train(self, x_train, y_train, x_val, y_val, epochs: int = 10, batch_size: int = 32, **fit_kwargs):

        if self.model is None:
            print("Model belum ada")
            print("Membuat model baru...")
            self.build_model(self._dropout_rate, self._learning_rate)

        self.history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2, **fit_kwargs)

        return self.history

    def evaluate(self, x_test, y_test):
        if self.model is None:
            raise RuntimeError("Model belum ada / bobot belum dimuat.")

        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(x_test, verbose=0), axis=1)
        y_true = y_test.squeeze()
        self.f1_macro = f1_score(y_true, y_pred, average="macro")

        print(f"Loss: {loss:.4f} | Acc: {acc:.4f} | Macro-F1: {self.f1_macro:.4f}")
        return self.f1_macro

    def save_weights(self, filename: str = "model.weights.h5"):
        if self.model is None:
            raise RuntimeError("Model belum ada.")
        
        path = self.w_dir / filename
        self.model.save_weights(path)
        print(f"Bobot disimpan di {path}")

    def load_weights(self, filename: str = "model.weights.h5"):

        path = self.w_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Tidak ada file bobot: {path}")
        if self.model is None:
            self.build_model(self._dropout_rate, self._learning_rate)

        self.model.load_weights(path)

        print(f"Bobot dimuat dari {path}")