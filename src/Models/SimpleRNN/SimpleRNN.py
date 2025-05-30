import os, pathlib
import numpy as np
import tensorflow as tf
from keras import Sequential, layers, optimizers
from sklearn.metrics import f1_score
from tqdm import tqdm
from Models.SimpleRNN.forward_layers_rnn import (
    embedding_forward, 
    rnn_layer_forward, 
    bidirectional_rnn_layer_forward,
    dense_forward, 
    dropout_forward
)


class SimpleRNNModel:
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_unit, num_classes: int, sequence_length: int, bidirectional=False, w_dir: str = "src/Models/SimpleRNN/training/weights"):
        self.vocab_size        = vocab_size
        self.embedding_dim     = embedding_dim

        # Kalau integer, jadiin list
        if isinstance(rnn_unit, int):
            self.rnn_unit = [rnn_unit]
        # Kalau list, tetep jadi list
        else:
            self.rnn_unit = list(rnn_unit)
        
        # Kalau boolean, jadiin list
        if isinstance(bidirectional, bool):
            self.bidirectional = [bidirectional] * len(self.rnn_unit)
        # Kalau list, tetep jadi list
        else:
            self.bidirectional = list(bidirectional)

        if len(self.bidirectional) != len(self.rnn_unit):
            raise ValueError(f"Panjang `bidirectional` ({len(self.bidirectional)}) dan `rnn_unit` ({len(self.rnn_unit)}) harus sama.")
            
        self.num_classes       = num_classes
        self.sequence_length   = sequence_length

        self.model     = None
        self.history   = None
        self.f1_macro  = None

        self.w_dir = pathlib.Path(w_dir)
        self.w_dir.mkdir(parents=True, exist_ok=True)

        self._dropout_rate  = 0.3
        self._learning_rate = 0.0001

    def build_model(self, dropout_rate : float = 0.3, learning_rate: float = 0.0001):
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

        for idx in range(len(self.rnn_unit)):
            units = self.rnn_unit[idx]
            b = self.bidirectional[idx]

            if idx < len(self.rnn_unit) - 1:
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

    def train(self, x_train, y_train, x_val, y_val, epochs: int = 10, batch_size: int = 128, **fit_kwargs):

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

    def load_layer_weights(self, layer_idx):
        layer = self.model.layers[layer_idx]
        weights = layer.get_weights()
        
        weights_float32 = []
        for w in weights:
            weights_float32.append(w.astype(np.float32))
        
        return layer, weights_float32
   
    def forward_scratch(self, input_sequences):
        if self.model is None:
            raise ValueError("Model belum ada")
        
        batch_size = input_sequences.shape[0]
        outputs = []
        
        for idx in tqdm(range(batch_size), desc="Forward Propagation", ncols=80):
            sequence = input_sequences[idx]
            
            layer_idx = 0
            
            # 1. Layer Embedding
            _, weights = self.load_layer_weights(layer_idx)
            matriks_embedding = weights[0] 
            
            embedded = embedding_forward(sequence, matriks_embedding)
            layer_idx += 1

            x = embedded
            
            # 2. Layer RNN
            for rnn_idx in range(len(self.rnn_unit)):
                layer, weights = self.load_layer_weights(layer_idx)
                
                if rnn_idx < len(self.rnn_unit) - 1:
                    return_sequences = True
                else:
                    return_sequences = False
                
                # Bidirectional 
                if self.bidirectional[rnn_idx]:
                    
                    # Forward 
                    W_input_hidden_maju = weights[0]   
                    W_hidden_hidden_maju = weights[1]  
                    b_maju = weights[2]                
                    
                    # Backward
                    W_input_hidden_mundur = weights[3]   
                    W_hidden_hidden_mundur = weights[4]  
                    b_mundur = weights[5]                
                    
                    hidden_size = self.rnn_unit[rnn_idx]
                    h_init_fwd = np.zeros(hidden_size, dtype=np.float32)
                    h_init_bwd = np.zeros(hidden_size, dtype=np.float32)
                    
                    x = bidirectional_rnn_layer_forward(
                        x, 
                        h_init_fwd, W_input_hidden_maju, W_hidden_hidden_maju, b_maju,
                        h_init_bwd, W_input_hidden_mundur, W_hidden_hidden_mundur, b_mundur,
                        merge_mode="concat",
                        return_sekuens=return_sequences
                    )
                else:
                    # Unidirectional
                    W_input_hidden = weights[0]   
                    W_hidden_hidden = weights[1]  
                    b = weights[2]                
                    
                    hidden_size = self.rnn_unit[rnn_idx]
                    h_init = np.zeros(hidden_size, dtype=np.float32)
                    
                    x = rnn_layer_forward(
                        x, h_init, W_input_hidden, W_hidden_hidden, b,
                        return_sekuens=return_sequences
                    )
                
                layer_idx += 1
            
            # 3. Layer Dropout
            x = dropout_forward(x)
            layer_idx += 1
            
            # 4. Layer Dense
            _, weights = self.load_layer_weights(layer_idx)
            W_dense = weights[0]
            b_dense = weights[1]
            
            output = dense_forward(x, W_dense, b_dense, fungsi_aktivasi="softmax")
            
            outputs.append(output)
        
        return np.array(outputs)