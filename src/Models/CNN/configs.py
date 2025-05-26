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

conv_layers_variations = {
    "1 Layer": {"n_conv_layers": 1, "filters_per_layer": [32], "kernel_sizes": [3], "pooling_type": "max", "n_epochs": 5},
    "2 Layers": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": 5},
    "3 Layers": {"n_conv_layers": 3, "filters_per_layer": [32, 64, 128], "kernel_sizes": [3, 3, 3], "pooling_type": "max", "n_epochs": 5},
}

filter_variations = {
    "Filters 16-32": {"n_conv_layers": 2, "filters_per_layer": [16, 32], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": 5},
    "Filters 32-64": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": 5},
    "Filters 64-128": {"n_conv_layers": 2, "filters_per_layer": [64, 128], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": 5},
}

kernel_size_variations = {
    "Kernel 3x3": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": 5},
    "Kernel 5x5": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [5, 5], "pooling_type": "max", "n_epochs": 5},
    "Kernel 7x7": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [7, 7], "pooling_type": "max", "n_epochs": 5},
}

pooling_variations = {
    "Max Pooling": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "max", "n_epochs": 5},
    "Avg Pooling": {"n_conv_layers": 2, "filters_per_layer": [32, 64], "kernel_sizes": [3, 3], "pooling_type": "average", "n_epochs": 5},
}
