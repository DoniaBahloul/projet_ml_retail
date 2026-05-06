"""
predict.py
==========
Module de prédiction pour de nouveaux clients.

Charge le pipeline sauvegardé (scaler + modèles) et retourne :
- Probabilité de churn
- Segment cluster KMeans
- Niveau de risque

Usage:
    python src/predict.py  (démonstration sur un client exemple)
    
    Ou depuis Flask :
    from src.predict import predict_customer
    result = predict_customer(customer_dict)
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Chemins ───────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

# ── Labels métier pour les clusters ──────────────────────────────────────────
CLUSTER_LABELS = {
    0: "Champions",
    1: "Clients réguliers",
    2: "Clients à risque",
    3: "Clients dormants",
}

CHURN_RISK_LEVELS = [
    (0.15, "Faible", "🟢", "Ce client est fidèle. Maintenez la relation."),
    (0.30, "Modéré", "🟡", "Surveillez ce client. Proposez une offre de fidélisation."),
    (0.45, "Élevé", "🟠", "Client à risque. Action marketing urgente recommandée."),
    (1.01, "Critique", "🔴", "Churn quasi-certain. Intervention immédiate nécessaire."),
]


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline():
    """Charge le scaler, le modèle churn, le modèle clustering et les feature names."""
    scaler_path = MODELS_DIR / "scaler.joblib"
    churn_path = MODELS_DIR / "churn_rf_model.joblib"
    kmeans_path = MODELS_DIR / "kmeans_model.joblib"
    features_path = MODELS_DIR / "feature_names.joblib"

    missing = [p for p in [scaler_path, churn_path, kmeans_path, features_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Modèles manquants : {missing}\n"
            "➡️  Exécutez d'abord : python src/preprocessing.py && python src/train_model.py"
        )

    scaler = joblib.load(scaler_path)
    churn_model = joblib.load(churn_path)
    kmeans = joblib.load(kmeans_path)
    feature_names = joblib.load(features_path)

    return scaler, churn_model, kmeans, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# PRÉPARATION DES FEATURES D'UN NOUVEAU CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_input(customer_raw: dict, feature_names: list[str], scaler) -> pd.DataFrame:
    """
    Convertit un dictionnaire de features brutes en DataFrame compatible
    avec le pipeline entraîné.
    Les features manquantes sont remplies avec la moyenne d'entraînement (scaler.mean_)
    pour être neutres après la normalisation StandardScaler.
    """
    df_input = pd.DataFrame([customer_raw])

    # Aligner sur les features attendues avec imputation de la moyenne
    for i, col in enumerate(feature_names):
        if col not in df_input.columns:
            df_input[col] = scaler.mean_[i]

    # Conserver uniquement les features connues, dans le bon ordre
    df_input = df_input[feature_names]
    return df_input


# ══════════════════════════════════════════════════════════════════════════════
# PRÉDICTION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def predict_customer(customer_raw: dict) -> dict:
    """
    Prédit le churn et le segment d'un nouveau client.

    Parameters
    ----------
    customer_raw : dict
        Dictionnaire avec les features numériques du client
        (après encodage ordinal basique).

    Returns
    -------
    dict avec :
        - churn_proba      : float [0,1]
        - churn_predicted  : int (0 ou 1)
        - risk_level       : str ("Faible", "Modéré", "Élevé", "Critique")
        - risk_emoji       : str
        - risk_message     : str
        - cluster_id       : int
        - cluster_label    : str
        - input_features   : dict (features utilisées)
    """
    scaler, churn_model, kmeans, feature_names = load_pipeline()

    # Préparer l'input
    X_input = preprocess_input(customer_raw, feature_names, scaler)
    X_scaled = scaler.transform(X_input)

    # Prédiction churn
    churn_proba = churn_model.predict_proba(X_scaled)[0][1]
    churn_predicted = int(churn_proba >= 0.5)

    # Niveau de risque
    for threshold, level, emoji, message in CHURN_RISK_LEVELS:
        if churn_proba < threshold:
            risk_level, risk_emoji, risk_message = level, emoji, message
            break

    # Cluster
    cluster_id = int(kmeans.predict(X_scaled)[0])
    cluster_label = CLUSTER_LABELS.get(cluster_id, f"Segment {cluster_id}")

    return {
        "churn_proba": round(float(churn_proba), 4),
        "churn_predicted": churn_predicted,
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "risk_message": risk_message,
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "churn_label": "Churn probable" if churn_predicted == 1 else ("À risque" if risk_level != "Faible" else "Client fidèle"),
        "input_features": X_input.iloc[0].to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DÉMONSTRATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Exemple de client à risque très élevé (Critique) - Basé sur données réelles du test set
    client_risque = {
        "Recency": 395.0,
        "Frequency": 13.0,
        "MonetaryTotal": 1760.0,
        "MonetaryAvg": 130.0,
        "MonetaryStd": 13.0,
        "CustomerTenure": 443.0,
        "Age": 65.0,
        "SupportTickets": 11.0,
        "Satisfaction": 4.0,
        "RFMSegment": 0,       # Dormants
        "AgeCategory": 5,      # 65+
        "SpendingCat": 0,      # Low
        "ChurnRisk": 2,        # Élevé
        "LoyaltyLevel": 3,     # Ancien
        "BasketSize": 0,       # Petit
        "PreferredTime": 3,    # Soir
        "WeekendRatio": 0.15,
        "ReturnRatio": 0.75,
        "UniqueProducts": 5,
        "CancelledTrans": 26,
    }

    # Exemple de client fidèle
    client_fidele = {
        "Recency": 10,
        "Frequency": 35,
        "MonetaryTotal": 8500.0,
        "MonetaryAvg": 242.0,
        "MonetaryStd": 80.0,
        "CustomerTenure": 600,
        "Age": 32,
        "SupportTickets": 1.0,
        "Satisfaction": 4.5,
        "RFMSegment": 3,       # Champions
        "AgeCategory": 1,      # 25-34
        "SpendingCat": 3,      # VIP
        "ChurnRisk": 0,        # Faible
        "LoyaltyLevel": 2,     # Établi
        "BasketSize": 2,       # Grand
        "PreferredTime": 3,    # Soir
        "WeekendRatio": 0.45,
        "ReturnRatio": 0.05,
        "UniqueProducts": 120,
        "CancelledTrans": 1,
    }

    print("\n" + "=" * 60)
    print("🔴 CLIENT À RISQUE :")
    print("=" * 60)
    try:
        r1 = predict_customer(client_risque)
        print(f"   Probabilité churn : {r1['churn_proba']:.1%}")
        print(f"   Niveau de risque  : {r1['risk_emoji']} {r1['risk_level']}")
        print(f"   Segment           : {r1['cluster_label']}")
        print(f"   Recommandation    : {r1['risk_message']}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")

    print("\n" + "=" * 60)
    print("🟢 CLIENT FIDÈLE :")
    print("=" * 60)
    try:
        r2 = predict_customer(client_fidele)
        print(f"   Probabilité churn : {r2['churn_proba']:.1%}")
        print(f"   Niveau de risque  : {r2['risk_emoji']} {r2['risk_level']}")
        print(f"   Segment           : {r2['cluster_label']}")
        print(f"   Recommandation    : {r2['risk_message']}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
