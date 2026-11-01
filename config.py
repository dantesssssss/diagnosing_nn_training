from pathlib import Path

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

INPUT_DIM = 28 * 28  # 784 features 28*28 pixels in mstfashion
SEED = 2026
BATCH_SIZE = 128
RESULTS_DIR = Path("../results")
OUTPUT_UNITS = 10
LEARNING_RATE = 1e-3


