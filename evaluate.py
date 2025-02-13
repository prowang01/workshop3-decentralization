import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Charger le dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir y en one-hot encoding
y_test_categorical = to_categorical(y_test, num_classes=3)

# Charger ton modèle
model = tf.keras.models.load_model("iris_nn_model.h5")

# Évaluer la précision
loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"Model accuracy on test data: {accuracy * 100:.2f}%")
