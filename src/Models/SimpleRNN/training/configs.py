N_EPOCHS_RNN = 10
EMBEDDING_DIM = 128

# 1. Variasi Jumlah Layer
VARIASI_LAYER_RNN = {
    "1Layer_128unit_uni": {
        "rnn_unit": [128],
        "bidirectional": [False],
        "n_epochs": N_EPOCHS_RNN
    },
    "2Layer_128unit_uni": {
        "rnn_unit": [128, 128],
        "bidirectional": [False, False],
        "n_epochs": N_EPOCHS_RNN
    },
    "3Layer_128unit_uni": {
        "rnn_unit": [128, 128, 128],
        "bidirectional": [False, False, False],
        "n_epochs": N_EPOCHS_RNN
    },
}

# 2. Variasi Banyak Cell
VARIASI_CELL_RNN = {
    "2Layer_64cell_uni": {
        "rnn_unit": [64, 64],
        "bidirectional": [False, False],
        "n_epochs": N_EPOCHS_RNN
    },
    "2Layer_128cell_uni": {
        "rnn_unit": [128, 128],
        "bidirectional": [False, False],
        "n_epochs": N_EPOCHS_RNN
    },
    "2Layer_256cell_uni": {
        "rnn_unit": [256, 256],
        "bidirectional": [False, False],
        "n_epochs": N_EPOCHS_RNN
    },
}

# 3. Variasi Arah  
VARIASI_ARAH_RNN = {
    "2Layer_128cell_uni": {
        "rnn_unit": [128, 128],
        "bidirectional": [False, False],
        "n_epochs": N_EPOCHS_RNN
    },
    "2Layer_128cell_bi": {
        "rnn_unit": [128, 128],
        "bidirectional": [True, True],
        "n_epochs": N_EPOCHS_RNN
    },
}

CONFIG_GABUNGAN = {
    "VariasiLayer": VARIASI_LAYER_RNN,
    "VariasiCell": VARIASI_CELL_RNN,
    "VariasiArah": VARIASI_ARAH_RNN,
}