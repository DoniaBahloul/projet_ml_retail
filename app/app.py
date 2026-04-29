"""
app.py
======
Application Flask pour l'interface de prédiction du churn client.

Routes :
  GET  /           → Page d'accueil avec formulaire
  POST /predict    → Effectue la prédiction et affiche les résultats
  GET  /dashboard  → Dashboard avec visualisations des modèles

Usage:
    cd projet_ml_retail
    python app/app.py
    
    Puis ouvrir : http://127.0.0.1:5000
"""

import os
import sys
from pathlib import Path

# Ajouter le dossier racine au path pour importer src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request

from src.predict import predict_customer

app = Flask(__name__)
app.secret_key = "retail_ml_2025"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Page d'accueil avec formulaire de saisie client."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Effectue la prédiction churn à partir du formulaire."""
    try:
        # Récupérer les données du formulaire
        form = request.form

        customer_data = {
            "Recency": float(form.get("Recency", 100)),
            "Frequency": float(form.get("Frequency", 5)),
            "MonetaryTotal": float(form.get("MonetaryTotal", 500)),
            "MonetaryAvg": float(form.get("MonetaryAvg", 100)),
            "MonetaryStd": float(form.get("MonetaryStd", 50)),
            "CustomerTenure": float(form.get("CustomerTenure", 180)),
            "Age": float(form.get("Age", 35)),
            "SupportTickets": float(form.get("SupportTickets", 2)),
            "Satisfaction": float(form.get("Satisfaction", 3)),
            "WeekendRatio": float(form.get("WeekendRatio", 0.3)),
            "ReturnRatio": float(form.get("ReturnRatio", 0.05)),
            "UniqueProducts": float(form.get("UniqueProducts", 20)),
            "CancelledTrans": float(form.get("CancelledTrans", 2)),
            # Encodages ordinaux simplifiés
            "RFMSegment": int(form.get("RFMSegment", 1)),
            "SpendingCat": int(form.get("SpendingCat", 1)),
            "ChurnRisk": int(form.get("ChurnRisk", 1)),
        }

        result = predict_customer(customer_data)
        return render_template("result.html", result=result, customer=customer_data)

    except FileNotFoundError as e:
        error_msg = str(e)
        return render_template("index.html", error=error_msg), 200

    except Exception as e:
        return render_template("index.html", error=f"Erreur de prédiction : {str(e)}"), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API REST — Retourne la prédiction en JSON."""
    try:
        data = request.get_json(force=True)
        result = predict_customer(data)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400



@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "ML Retail Churn Predictor"})


# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Démarrage de l'application Flask...")
    print("   URL : http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
