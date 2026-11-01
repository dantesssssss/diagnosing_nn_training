import data
import model_factory
from config import RESULTS_DIR
from plots import plot_compare

X_train_s, X_val_s, _, y_train, y_val, _ = data.load_data()

# model with 2 neurons
model = model_factory.build_model(layers=1, dense_units=2)
history_a = model_factory.train(name="underfit_A_2_neurons", model=model, epochs=30, X_train=X_train_s, y_train=y_train,
                                X_val=X_val_s,
                                y_val=y_val)

# model with 16 neurons
model = model_factory.build_model(layers=1, dense_units=16)
history_b = model_factory.train(name="underfit_B_16_neurons", model=model, epochs=30, X_train=X_train_s,
                                y_train=y_train,
                                X_val=X_val_s,
                                y_val=y_val)

# model with 256 neurons
model = model_factory.build_model(layers=1, dense_units=256)

history_c = model_factory.train(name="underfit_C_256_neurons", model=model, epochs=30, X_train=X_train_s,
                                y_train=y_train,
                                X_val=X_val_s,
                                y_val=y_val)

plot_compare(
    histories={"A: 2 neurons": history_a, "B: 16 neurons": history_b, "C: 256 neurons": history_c},
    title="Underfitting 2,16,256 : 50,000 images",
    path=RESULTS_DIR / "underfitting_compare.png",
)
