from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tensorflow.keras.callbacks import EarlyStopping

import data
import model_factory
from config import CLASS_NAMES, RESULTS_DIR
from plots import plot_confusion

output_dir = RESULTS_DIR / "final_model"

X_train_s, X_val_s, X_test_s, y_train, y_val, y_test = data.load_data()

# final model
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)

model = model_factory.build_model(layers=3, dense_units=256, dropout=0.3)
history = model_factory.train(name="final_model", model=model, epochs=100, X_train=X_train_s, y_train=y_train,
                              X_val=X_val_s,
                              y_val=y_val,
                              callbacks=[early_stopping])

# results of final model
probs = model.predict(X_test_s, verbose=0)
y_pred = probs.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3)
with open(output_dir / "test_metrics.txt", "w") as f:
    f.write(f"accuracy {accuracy:.4f}\nmacro precision {precision}\nrecall {recall}\nf1 {f1}\n")
    f.write(report)
plot_confusion(y_true=y_test, y_pred=y_pred, title="Final model: test set", path=output_dir / "confusion_matrix.png")
