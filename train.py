import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Charger le dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_categorical = to_categorical(y_train, num_classes=3)

# Fonction pour créer et entraîner un modèle avec des variations
def create_and_train_model(random_seed, filename):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    model = Sequential([
        Dense(10, activation='relu', input_shape=(4,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_categorical, epochs=50, batch_size=10, verbose=0)

    # Sauvegarder le modèle
    model.save(filename)
    print(f"Modèle sauvegardé sous {filename}")

# Entraîner et sauvegarder 3 modèles différents
create_and_train_model(42, "model_1.h5")
create_and_train_model(84, "model_2.h5")
create_and_train_model(126, "model_3.h5")
