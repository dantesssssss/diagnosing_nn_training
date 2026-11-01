from tensorflow.keras.callbacks import EarlyStopping

import data
import model_factory
from config import RESULTS_DIR, SEED
from plots import plot_compare

X_small, X_val_s, _, y_small, y_val, _ = data.load_data(2000)

print("Small training set:", len(X_small))

# model with 512 neurons per layer, with small set of data to learn
model = model_factory.build_model(layers=3, dense_units=512)
history_a = model_factory.train(name="overfitted_model_dropout_0", model=model, epochs=100, X_train=X_small,
                                y_train=y_small,
                                X_val=X_val_s,
                                y_val=y_val)
# model with 512 neurons per layer, with small set of data to learn, dropout=0.5
model = model_factory.build_model(layers=3, dense_units=512, dropout=0.5)
history_b = model_factory.train(name="overfitted_model_dropout_0.5", model=model, epochs=100, X_train=X_small,
                                y_train=y_small,
                                X_val=X_val_s,
                                y_val=y_val)

# model with 512 neurons per layer, with small set of data to learn early stopping
model = model_factory.build_model(layers=3, dense_units=512, dropout=0.5)

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
history_c = model_factory.train(name="overfitted_model_dropout_0.5_ES", model=model, epochs=100, X_train=X_small,
                                y_train=y_small,
                                X_val=X_val_s,
                                y_val=y_val,
                                callbacks=[early_stopping])

plot_compare(
    histories={"A: no regularization": history_a, "B: dropout 0.5": history_b,
               "C: dropout 0.5 + early stop": history_c},
    title="Overfitting: 2,000 images, 3x512 network",
    path=RESULTS_DIR / "overfitting_compare.png",
)
