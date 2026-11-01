import data
import model_factory
from config import RESULTS_DIR
from plots import plot_compare

DROPOUT_RATES = [0.0, 0.3, 0.5, 0.7]
histories = {}
X_train_s, X_val_s, _, y_train, y_val, _ = data.load_data()



# models with different dropout
for rate in DROPOUT_RATES:
    model = model_factory.build_model(layers=3, dense_units=256,dropout=rate)
    histories[f"dropout_{rate}"] = model_factory.train(name=f"dropout_{rate}", model=model, epochs=30, X_train=X_train_s, y_train=y_train,
                                  X_val=X_val_s,
                                  y_val=y_val)

plot_compare(histories=histories,title="Experiment: dropout probability (3x256, 50,000 images)",
             path=RESULTS_DIR / "dropout_experiment.png")

