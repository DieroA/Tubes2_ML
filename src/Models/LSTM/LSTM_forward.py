import numpy as np

class LSTMScratch:
    """
    Implementasi LSTM dari scratch untuk forward propagation saja.
    Dirancang untuk kompatibel dengan weights dari model Keras.
    """
    
    def __init__(self, keras_model):
        """
        Inisialisasi dengan model Keras yang sudah trained.
        
        Args:
            keras_model: Model Keras yang sudah di-train
        """
        self.model = keras_model
        self.params = self._extract_keras_weights()
        
    def _extract_keras_weights(self):
        """Ekstrak weights dari model Keras ke format yang bisa digunakan"""
        params = {}
        
        embedding_layer = self.model.get_layer('embedding')
        params['embedding'] = embedding_layer.get_weights()[0]
        
        lstm_num_layers = 0
        for i, layer in enumerate(self.model.layers):
            if 'lstm' in layer.name.lower() or 'bidirectional' in layer.name.lower():
                if 'bidirectional' in layer.name.lower():
                    lstm_layer = layer.forward_layer
                    is_bidirectional = True
                else:
                    lstm_layer = layer
                    is_bidirectional = False
                
                weights = lstm_layer.get_weights()
                
                prefix = f'lstm_{lstm_num_layers+1}'
                
                params[f'{prefix}_Wi'] = weights[0][:, :lstm_layer.units]
                params[f'{prefix}_Wf'] = weights[0][:, lstm_layer.units:2*lstm_layer.units]
                params[f'{prefix}_Wc'] = weights[0][:, 2*lstm_layer.units:3*lstm_layer.units]
                params[f'{prefix}_Wo'] = weights[0][:, 3*lstm_layer.units:]
                
                params[f'{prefix}_Ui'] = weights[1][:, :lstm_layer.units]
                params[f'{prefix}_Uf'] = weights[1][:, lstm_layer.units:2*lstm_layer.units]
                params[f'{prefix}_Uc'] = weights[1][:, 2*lstm_layer.units:3*lstm_layer.units]
                params[f'{prefix}_Uo'] = weights[1][:, 3*lstm_layer.units:]
                
                params[f'{prefix}_bi'] = weights[2][:lstm_layer.units]
                params[f'{prefix}_bf'] = weights[2][lstm_layer.units:2*lstm_layer.units]
                params[f'{prefix}_bc'] = weights[2][2*lstm_layer.units:3*lstm_layer.units]
                params[f'{prefix}_bo'] = weights[2][3*lstm_layer.units:]
                
                if is_bidirectional:
                    weights_bw = layer.backward_layer.get_weights()
                    params[f'{prefix}_Wi_bw'] = weights_bw[0][:, :lstm_layer.units]
                    params[f'{prefix}_Wf_bw'] = weights_bw[0][:, lstm_layer.units:2*lstm_layer.units]
                    params[f'{prefix}_Wc_bw'] = weights_bw[0][:, 2*lstm_layer.units:3*lstm_layer.units]
                    params[f'{prefix}_Wo_bw'] = weights_bw[0][:, 3*lstm_layer.units:]
                    
                    params[f'{prefix}_Ui_bw'] = weights_bw[1][:, :lstm_layer.units]
                    params[f'{prefix}_Uf_bw'] = weights_bw[1][:, lstm_layer.units:2*lstm_layer.units]
                    params[f'{prefix}_Uc_bw'] = weights_bw[1][:, 2*lstm_layer.units:3*lstm_layer.units]
                    params[f'{prefix}_Uo_bw'] = weights_bw[1][:, 3*lstm_layer.units:]
                    
                    params[f'{prefix}_bi_bw'] = weights_bw[2][:lstm_layer.units]
                    params[f'{prefix}_bf_bw'] = weights_bw[2][lstm_layer.units:2*lstm_layer.units]
                    params[f'{prefix}_bc_bw'] = weights_bw[2][2*lstm_layer.units:3*lstm_layer.units]
                    params[f'{prefix}_bo_bw'] = weights_bw[2][3*lstm_layer.units:]
                lstm_num_layers += 1

        classifier_layer = self.model.get_layer('classifier')
        params['W_classifier'] = classifier_layer.get_weights()[0]
        params['b_classifier'] = classifier_layer.get_weights()[1]
        
        return params
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def lstm_cell_forward(self, xt, h_prev, c_prev, params, prefix, bidirectional=False):
        """
        Implementasi single LSTM cell forward pass.
        
        Args:
            xt: Input at timestep t
            h_prev: Hidden state at previous timestep
            c_prev: Cell state at previous timestep
            params: Dictionary berisi parameters
            prefix: Prefix untuk parameter names (e.g., 'lstm_1')
            bidirectional: Apakah ini backward pass untuk bidirectional
            
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        suffix = '_bw' if bidirectional else ''
        
        i = self.sigmoid(
            np.dot(xt, params[f'{prefix}_Wi{suffix}']) + 
            np.dot(h_prev, params[f'{prefix}_Ui{suffix}']) + 
            params[f'{prefix}_bi{suffix}']
        )
        
        f = self.sigmoid(
            np.dot(xt, params[f'{prefix}_Wf{suffix}']) + 
            np.dot(h_prev, params[f'{prefix}_Uf{suffix}']) + 
            params[f'{prefix}_bf{suffix}']
        )
        
        o = self.sigmoid(
            np.dot(xt, params[f'{prefix}_Wo{suffix}']) + 
            np.dot(h_prev, params[f'{prefix}_Uo{suffix}']) + 
            params[f'{prefix}_bo{suffix}']
        )
        
        c_tilde = self.tanh(
            np.dot(xt, params[f'{prefix}_Wc{suffix}']) + 
            np.dot(h_prev, params[f'{prefix}_Uc{suffix}']) + 
            params[f'{prefix}_bc{suffix}']
        )
        
        c_next = f * c_prev + i * c_tilde
        
        h_next = o * self.tanh(c_next)
        
        return h_next, c_next
    
    def lstm_forward(self, x, params, layer_idx, bidirectional=False):
        """
        Forward pass untuk satu LSTM layer.
        
        Args:
            x: Input data (sequence_length, batch_size, input_dim)
            params: Dictionary berisi parameters
            layer_idx: Index layer saat ini
            bidirectional: Apakah layer ini bidirectional
            
        Returns:
            h: Hidden states untuk semua timesteps
            c: Cell states untuk semua timesteps
        """
        prefix = f'lstm_{layer_idx+1}'
        sequence_length, batch_size, _ = x.shape
        units = params[f'{prefix}_Wi'].shape[1]
        
        h = np.zeros((sequence_length, batch_size, units))
        c = np.zeros((sequence_length, batch_size, units))
        
        h_prev = np.zeros((batch_size, units))
        c_prev = np.zeros((batch_size, units))
        
        if bidirectional:
            for t in range(sequence_length):
                h_next, c_next = self.lstm_cell_forward(
                    x[t], h_prev, c_prev, params, prefix, bidirectional=False
                )
                h[t] = h_next
                c[t] = c_next
                h_prev = h_next
                c_prev = c_next
            
            h_bw = np.zeros((sequence_length, batch_size, units))
            c_bw = np.zeros((sequence_length, batch_size, units))
            
            h_prev_bw = np.zeros((batch_size, units))
            c_prev_bw = np.zeros((batch_size, units))
            
            for t in reversed(range(sequence_length)):
                h_next_bw, c_next_bw = self.lstm_cell_forward(
                    x[t], h_prev_bw, c_prev_bw, params, prefix, bidirectional=True
                )
                h_bw[t] = h_next_bw
                c_bw[t] = c_next_bw
                h_prev_bw = h_next_bw
                c_prev_bw = c_next_bw
            
            h = np.concatenate((h, h_bw), axis=-1)
        else:
            for t in range(sequence_length):
                h_next, c_next = self.lstm_cell_forward(
                    x[t], h_prev, c_prev, params, prefix, bidirectional=False
                )
                h[t] = h_next
                c[t] = c_next
                h_prev = h_next
                c_prev = c_next
        
        return h
    
    def forward_propagation(self, X):
        """
        Forward propagation melalui seluruh network.
        
        Args:
            X: Input data (batch_size, sequence_length)
            
        Returns:
            output: Prediksi akhir
        """
        batch_size = X.shape[0]
        sequence_length = X.shape[1]
        
        embedded = np.zeros((batch_size, sequence_length, self.params['embedding'].shape[1]))
        for i in range(batch_size):
            for t in range(sequence_length):
                embedded[i, t] = self.params['embedding'][X[i, t]]
        
        embedded = embedded.transpose(1, 0, 2)
        
        h = embedded
        
        for i in range(len([l for l in self.model.layers if 'lstm' in l.name.lower() or 'bidirectional' in l.name.lower()])):
            bidirectional = 'bidirectional' in self.model.layers[i].name.lower()
            h = self.lstm_forward(h, self.params, i, bidirectional=bidirectional)
            
            if i < len([l for l in self.model.layers if 'lstm' in l.name.lower() or 'bidirectional' in l.name.lower()]) - 1:
                h = h[-1]  
                h = h[np.newaxis, :, :]  
            else:
                if not bidirectional:
                    h = h[-1]  
                else:
                    h = h[-1]
        
        output = self.softmax(np.dot(h, self.params['W_classifier']) + self.params['b_classifier'])
        
        return output
    
    def predict(self, X):
        """
        Prediksi class untuk input X.
        
        Args:
            X: Input data (batch_size, sequence_length)
            
        Returns:
            Predicted classes
        """
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)