import numpy as np

def tanh(x):
    return np.tanh(x)

def softmax(x):
    maksimum = np.max(x, axis=-1, keepdims=True)
    nilai_shift = x - maksimum 
    e_x = np.exp(nilai_shift)
    # print("e_x:", e_x)
    jumlah = np.sum(e_x, axis=-1, keepdims=True)
    # print("jumlah:", jumlah)
    return e_x / jumlah

def linear(x):
    return x

def embedding_forward(sekuens_token, matriks_embedding):
    vocab_size = matriks_embedding.shape[0]
    clip = np.clip(sekuens_token, 0, vocab_size - 1)
    return matriks_embedding[clip]

def rnn_cell_forward(x_t, h_sebelum, W_input_hiden, W_hiden_hiden, b, fungsi_aktivasi=tanh):
    h_next = fungsi_aktivasi(np.dot(x_t, W_input_hiden) + np.dot(h_sebelum, W_hiden_hiden) + b)
    return h_next

def rnn_layer_forward(sekuens_input, h_awal, W_input_hiden, W_hiden_hiden, b, fungsi_aktivasi=tanh, return_sekuens=False):
    panjang = sekuens_input.shape[0]
    h_sekarang = h_awal.copy()

    if return_sekuens:
        all_output = []
        for t in range(panjang):
            x_t = sekuens_input[t, :]
            h_sekarang = rnn_cell_forward(x_t, h_sekarang, W_input_hiden, W_hiden_hiden, b, fungsi_aktivasi)
            all_output.append(h_sekarang.copy()) 
        return np.array(all_output)
    else:
        for t in range(panjang):
            x_t = sekuens_input[t, :]
            h_sekarang = rnn_cell_forward(x_t, h_sekarang, W_input_hiden, W_hiden_hiden, b, fungsi_aktivasi)
        return h_sekarang

def dense_forward(vektor_input, W_dense, b, fungsi_aktivasi="linear"):
    output = np.dot(vektor_input, W_dense) + b
    
    if fungsi_aktivasi == "softmax":
        return softmax(output)
    elif fungsi_aktivasi == "tanh":
        return tanh(output)
    else: 
        return linear(output)

def dropout_forward(input_tensor):
    return input_tensor

def bidirectional_rnn_layer_forward(sekuens_input, 
    h_awal_maju, W_input_hiden_maju, W_hiden_hiden_maju, b_h_maju, 
    h_awal_mundur, W_input_hiden_mundur, W_hiden_hiden_mundur, b_h_mundur, 
    fungsi_aktivasi=tanh, 
    merge_mode="concat",
    return_sekuens=False
):
    sekuens_input_copy = sekuens_input.copy()
    
    if return_sekuens:
        h_fwd_states = rnn_layer_forward(sekuens_input_copy, h_awal_maju.copy(), W_input_hiden_maju, W_hiden_hiden_maju, b_h_maju, fungsi_aktivasi, return_sekuens=True)
    else:
        h_fwd_final = rnn_layer_forward(sekuens_input_copy, h_awal_maju.copy(), W_input_hiden_maju, W_hiden_hiden_maju, b_h_maju, fungsi_aktivasi, return_sekuens=False)

    sekuens_input_reversed = sekuens_input_copy[::-1, :] 
    if return_sekuens:
        h_bwd_states_reversed = rnn_layer_forward(sekuens_input_reversed, h_awal_mundur.copy(), W_input_hiden_mundur, W_hiden_hiden_mundur, b_h_mundur, fungsi_aktivasi, return_sekuens=True)
        h_bwd_states = h_bwd_states_reversed[::-1, :] 
    else:
        h_bwd_final = rnn_layer_forward(sekuens_input_reversed, h_awal_mundur.copy(), W_input_hiden_mundur, W_hiden_hiden_mundur, b_h_mundur, fungsi_aktivasi, return_sekuens=False)

    if merge_mode == "concat":
        if return_sekuens:
            return np.concatenate((h_fwd_states, h_bwd_states), axis=-1)
        else:
            return np.concatenate((h_fwd_final, h_bwd_final), axis=-1)
    elif merge_mode == "sum":
        if return_sekuens:
            return h_fwd_states + h_bwd_states
        else:
            return h_fwd_final + h_bwd_final
    else:
        raise ValueError(f"Merge mode '{merge_mode}' tidak didukung.")