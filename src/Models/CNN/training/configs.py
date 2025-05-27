"""
CNN Config format: 
    "Name (Optional)" : {
        "n_conv_layers": x,
        "filters_per_layer": [x, y, ...],
        "kernel_sizes": [x, y, ...],
        "pooling_type": "max" or "average",
        "n_epochs": x
    }
"""

N_EPOCHS = 7

CONV_LAYER_VARIATIONS = {
    "1 Layer": {"n_conv_layers": 1, "filters_per_layer": [32], "kernel_sizes": [3], "pooling_type": "max", "n_epochs": N_EPOCHS},
    "2 Layers": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": N_EPOCHS},
    "3 Layers": {"n_conv_layers": 3, "filters_per_layer": [32, 64, 128], "kernel_sizes": [3, 3, 3], "pooling_type": "max", "n_epochs": N_EPOCHS},
}

FILTER_VARIATIONS = {
    "Filters 16-32": {"n_conv_layers": 2, "filters_per_layer": [16, 32], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": N_EPOCHS},
    "Filters 32-64": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": N_EPOCHS},
    "Filters 64-128": {"n_conv_layers": 2, "filters_per_layer": [64, 128], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": N_EPOCHS},
}

KERNEL_SIZE_VARIATIONS = {
    "Kernel 3x3": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": N_EPOCHS},
    "Kernel 5x5": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [5, 5], "pooling_type": "max", "n_epochs": N_EPOCHS},
    "Kernel 7x7": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [7, 7], "pooling_type": "max", "n_epochs": N_EPOCHS},
}

POOLING_VARIATIONS = {
    "Max Pooling": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": N_EPOCHS},
    "Avg Pooling": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "average", "n_epochs": N_EPOCHS},
}
