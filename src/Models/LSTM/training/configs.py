"""
LSTM Config format:
    "Name (Optional)": {
        "units": [x, y, ...],
        "bidirectional": [True/False, ...],  # Per layer
    }
"""


# Variasi jumlah layer LSTM
LAYER_VARIATIONS = {
    "1 Layer LSTM": {
        "units": [64],
        "bidirectional": [False],
    },
    "2 Layer LSTM": {
        "units": [64, 32],
        "bidirectional": [False, False],
    },
    "3 Layer LSTM": {
        "units": [64, 32, 16],
        "bidirectional": [False, False, False],
    }
}

# Variasi jumlah unit/cell per layer
UNIT_VARIATIONS = {
    "32 Units": {
        "units": [32, 32],
        "bidirectional": [False, False],
    },
    "64 Units": {
        "units": [64, 64],
        "bidirectional": [False, False],
    },
    "128 Units": {
        "units": [128, 128],
        "bidirectional": [False, False],
    }
}

# Variasi jenis layer (unidirectional vs bidirectional)
DIRECTION_VARIATIONS = {
    "Unidirectional": {
        "units": [64, 32],
        "bidirectional": [False, False],
    },
    "Bidirectional": {
        "units": [64, 32],
        "bidirectional": [True, True],
    },
    "Mixed Direction": {
        "units": [64, 32],
        "bidirectional": [True, False],  # First layer bidirectional, second unidirectional
    }
}

# Variasi dropout rate
DROPOUT_VARIATIONS = {
    "Low Dropout (0.2)": {
        "units": [64, 32],
        "bidirectional": [False, False],
    },
    "Medium Dropout (0.5)": {
        "units": [64, 32],
        "bidirectional": [False, False],
    },
    "High Dropout (0.8)": {
        "units": [64, 32],
        "bidirectional": [False, False],
    }
}