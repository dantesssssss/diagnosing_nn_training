from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

import plots
from config import SEED, RESULTS_DIR


def load_data(train_size=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (X_full, y_full), (X_test, y_test) = fashion_mnist.load_data()
    X_full = X_full.reshape(len(X_full), -1)
    X_test = X_test.reshape(len(X_test), -1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=10_000, stratify=y_full, random_state=SEED,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype("float32")
    X_val_s = scaler.transform(X_val).astype("float32")
    X_test_s = scaler.transform(X_test).astype("float32")

    print("Training / validation / test:", len(X_train), len(X_val), len(X_test))
    plots.plot_samples(X_train=X_train, y_train=y_train)

    if train_size is not None:
        X_train_s, _, y_train, _ = train_test_split(
            X_train_s, y_train, train_size=train_size, stratify=y_train, random_state=SEED,
        )
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test
