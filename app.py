from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
import os

# Charger les modèles
models = {
    "model_1": tf.keras.models.load_model("model_1.h5"),
    "model_2": tf.keras.models.load_model("model_2.h5"),
    "model_3": tf.keras.models.load_model("model_3.h5")
}

# Chemin de la base de données JSON
DB_PATH = "database.json"

# Charger les données du fichier JSON
def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as file:
            return json.load(file)
    return {"models": {}}

# Sauvegarder les données du fichier JSON
def save_database(data):
    with open(DB_PATH, "w") as file:
        json.dump(data, file, indent=4)

app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Prediction API with Slashing is running!"

@app.route("/predict", methods=["GET"])
def predict():
    try:
        db = load_database()  # Charger la base de données

        required_params = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

        # Vérifier si tous les paramètres sont présents
        for param in required_params:
            if param not in request.args:
                return jsonify({"status": "error", "message": f"Missing parameter: {param}"}), 400

        # Extraire les features
        features = np.array([[float(request.args[param]) for param in required_params]])

        predictions = {}
        penalties = {}

        # Faire une prédiction avec chaque modèle
        for model_name, model in models.items():
            
            # Vérifier si le modèle a encore un solde positif
            if db["models"][model_name]["balance"] <= 0:
                continue  # Ignorer ce modèle s'il est ruiné

            pred = model.predict(features).tolist()[0]
            predictions[model_name] = pred

            # Calculer l'erreur (différence avec la prédiction moyenne)
            avg_prediction = np.mean(list(predictions.values()), axis=0)
            predicted_class = int(np.argmax(avg_prediction))

            model_accuracy = db["models"].get(model_name, {}).get("accuracy", 1)
            error = np.linalg.norm(np.array(pred) - avg_prediction)

            # Slashing : pénaliser si le modèle est très loin de la prédiction moyenne
            penalty = 50 if error > 0.1 else 0  # Ex : Si erreur > 0.2, -50€
            penalties[model_name] = penalty
            db["models"][model_name]["balance"] -= penalty  # Réduction du solde

        # Sauvegarder les nouvelles balances
        save_database(db)

        return jsonify({
            "status": "success",
            "predictions": predictions,
            "final_prediction": {
                "class": predicted_class,
                "probability": float(avg_prediction[predicted_class])
            },
            "penalties": penalties,
            "balances": {m: db["models"][m]["balance"] for m in models}
        })

    except ValueError:
        return jsonify({"status": "error", "message": "Invalid parameter type."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
