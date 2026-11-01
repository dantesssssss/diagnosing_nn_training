from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, AdamW

import data
import model_factory
from config import RESULTS_DIR
from plots import plot_compare

def make_optimizers():
    return {
            "SGD": SGD(learning_rate=0.01),
            "SGD_momentum": SGD(learning_rate=0.01, momentum=0.9),
            "Adam": Adam(learning_rate=1e-3),
            "AdamW": AdamW(learning_rate=1e-3, weight_decay=0.004),
        }
histories = {}
X_train_s, X_val_s, _, y_train, y_val, _ = data.load_data()



# models with different optimizers
for name,optimizer in make_optimizers().items():
    model = model_factory.build_model(layers=3, dense_units=256,optimizer=optimizer)
    histories[f"optimizer_{name}"] = model_factory.train(name=f"optimizer_{name}", model=model, epochs=30, X_train=X_train_s, y_train=y_train,
                                  X_val=X_val_s,
                                  y_val=y_val)

plot_compare(histories=histories,title="Experiment: optimizer",
             path=RESULTS_DIR / "optimizer_experiment.png")


histories = {}

# same but with dropout and early stopping to avoid overfitting
for name,optimizer in make_optimizers().items():
    model = model_factory.build_model(layers=3, dense_units=256,dropout=0.3,optimizer=optimizer)
    histories[f"optimizer_{name}_dropout"] = model_factory.train(name=f"optimizer_{name}_dropout", model=model, epochs=100, X_train=X_train_s, y_train=y_train,
                                  X_val=X_val_s,
                                  y_val=y_val,
                                  callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)])

plot_compare(histories=histories,title="Experiment: optimizer_dropout",
             path=RESULTS_DIR / "optimizer_dropout_experiment.png")
