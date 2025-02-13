import requests
import time

# URL de l'API (remplace avec ton lien ngrok si besoin)
API_URL = "http://127.0.0.1:5000/predict"

# Générer des valeurs extrêmes pour forcer le slashing
extreme_values = [
    {"SepalLength": 10.0, "SepalWidth": 10.0, "PetalLength": 10.0, "PetalWidth": 10.0},
    {"SepalLength": -5.0, "SepalWidth": -3.4, "PetalLength": -1.5, "PetalWidth": -0.2},
    {"SepalLength": 20.0, "SepalWidth": 15.0, "PetalLength": 5.0, "PetalWidth": 2.0},
    {"SepalLength": 100.0, "SepalWidth": 50.0, "PetalLength": 30.0, "PetalWidth": 10.0},
]

# Nombre de requêtes à envoyer
num_requests = 50

print(f"🔥 Envoi de {num_requests} requêtes pour ruiner un modèle...")

for i in range(num_requests):
    params = extreme_values[i % len(extreme_values)]  # Tourne entre les valeurs extrêmes
    response = requests.get(API_URL, params=params)
    
    # Affiche la réponse
    try:
        data = response.json()
        print(f"Requête {i+1}/{num_requests} - Penalties: {data['penalties']} - Balances: {data['balances']}")
    except:
        print(f"Requête {i+1}/{num_requests} - Erreur")

    time.sleep(0.2)  # Petite pause pour éviter un blocage

print("✅ Test terminé ! Vérifie `database.json`.")
