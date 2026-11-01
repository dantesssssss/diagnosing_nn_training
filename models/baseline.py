import data
import model_factory

X_train_s, X_val_s, X_test_s, y_train, y_val, y_test = data.load_data()

# baseline model
model = model_factory.build_model(layers=3, dense_units=256)
history = model_factory.train(name="baseline_model", model=model, epochs=30, X_train=X_train_s, y_train=y_train,
                              X_val=X_val_s,
                              y_val=y_val)
