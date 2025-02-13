# workshop3-decentralization
Data Redundancy and Distributed Computing Workshop


## Structure des fichiers
- `app.py` → API Flask qui gère les prédictions, le consensus entre modèles, le slashing des modèles imprécis et la gestion des balances.
- `train.py` → Entraînement de plusieurs modèles distincts (`model_1.h5`, `model_2.h5`, etc.) pour simuler un environnement distribué.
- `evaluate.py` → Vérification de la précision des modèles sur un dataset de test.
- `spam_requests.py` → Script pour forcer le slashing en envoyant un grand nombre de requêtes extrêmes.
- `database.json` → Stocke les soldes des modèles et leurs performances.
- `requirements.txt` → Liste des dépendances nécessaires.

## Travail de groupe réalisé en autonomie
On devait comparer les modèles en groupe, en travaillant tous sur le même dataset, mais j'ai remarqué que l'on pouvait tester plusieurs modèles nous-mêmes pour ne pas être freiné par la progression des autres.

Bien entendu, pour garantir que l'approche distribuée fonctionne bien, l'API a été **testée avec `ngrok`**, permettant une exposition en réseau et une simulation d'un environnement multi-utilisateurs.

