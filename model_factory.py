import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import losses
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import set_random_seed

import plots
from config import INPUT_DIM, SEED, OUTPUT_UNITS, LEARNING_RATE, BATCH_SIZE, RESULTS_DIR


def build_model(*, layers: int, dense_units: int, dropout: float | int = 0, optimizer=None):
    layers_list = [Input(shape=(INPUT_DIM,))]
    clear_session()
    set_random_seed(SEED)
    if optimizer is None:
        optimizer = Adam(learning_rate=LEARNING_RATE)

    for i in range(layers):
        layers_list.append(Dense(
            units=dense_units,
            activation="relu",
            kernel_initializer=HeNormal(seed=SEED + i),
        ))
        if dropout > 0:
            layers_list.append(Dropout(dropout, seed=SEED + i))

    layers_list.append(Dense(
        OUTPUT_UNITS, activation="softmax",
        kernel_initializer=GlorotUniform(seed=2029)
    ))

    model = Sequential(layers_list)
    print("Number of parameters:", model.count_params())
    model.compile(loss=losses.SparseCategoricalCrossentropy(), optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def train(*, name, model, epochs, X_train, y_train, X_val, y_val, callbacks=()):
    run_dir = RESULTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=[CSVLogger(run_dir / "history.csv")] + list(callbacks),
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    metrics = {
        "epochs_run": len(history.history["loss"]),
        "train_loss": train_loss, "train_acc": train_acc,
        "val_loss": val_loss, "val_acc": val_acc,
        "gap": train_acc - val_acc,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    plots.plot_history(history, name, run_dir / "loss.png")
    return history
