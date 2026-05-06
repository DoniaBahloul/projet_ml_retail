import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix

def validate():
    # Chemins
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/train_test")
    
    print("🔍 Chargement des données et des modèles...")
    
    # Chargement
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    model = joblib.load(MODELS_DIR / "churn_rf_model.joblib")
    
    # Transformation
    X_test_scaled = scaler.transform(X_test)
    
    # Prédiction
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Création du DataFrame de comparaison
    results = pd.DataFrame({
        'Index': X_test.index,
        'Réel (Churn)': y_test,
        'Prédit (Churn)': y_pred,
        'Probabilité (%)': (y_proba * 100).round(2)
    })
    
    # Identifier Vrai/Faux
    # Vrai Positif (VP), Vrai Négatif (VN), Faux Positif (FP), Faux Négatif (FN)
    results['Résultat'] = results.apply(
        lambda x: "✅ VRAI" if x['Réel (Churn)'] == x['Prédit (Churn)'] else "❌ FAUX", 
        axis=1
    )
    
    # Détail du type d'erreur
    def get_type(row):
        if row['Réel (Churn)'] == 1 and row['Prédit (Churn)'] == 1: return "Vrai Positif (Churn détecté)"
        if row['Réel (Churn)'] == 0 and row['Prédit (Churn)'] == 0: return "Vrai Négatif (Fidèle détecté)"
        if row['Réel (Churn)'] == 0 and row['Prédit (Churn)'] == 1: return "Faux Positif (Fausse alerte)"
        if row['Réel (Churn)'] == 1 and row['Prédit (Churn)'] == 0: return "Faux Négatif (Churn manqué)"
        return ""
    
    results['Type'] = results.apply(get_type, axis=1)
    
    print("\n" + "="*80)
    print("📋 EXTRAIT DES RÉSULTATS (20 PREMIERS CLIENTS)")
    print("="*80)
    print(results.head(20).to_string(index=False))
    
    # Sauvegarder les erreurs pour analyse
    errors = results[results['Résultat'] == "❌ FAUX"]
    errors.to_csv("data/processed/errors.csv", index=False)
    print(f"\n📂 {len(errors)} erreurs sauvegardées dans data/processed/errors.csv pour analyse.")
    
    print("\n" + "="*80)
    print("📊 STATISTIQUES GLOBALES")
    print("="*80)
    
    total = len(results)
    vrais = (results['Résultat'] == "✅ VRAI").sum()
    faux = (results['Résultat'] == "❌ FAUX").sum()
    precision = (vrais / total) * 100
    
    print(f"Total testés : {total}")
    print(f"Total Vrais  : {vrais} ({(vrais/total)*100:.1f}%)")
    print(f"Total Faux   : {faux} ({(faux/total)*100:.1f}%)")
    print(f"Précision    : {precision:.2f}%")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion :")
    print(f"VN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | VP: {cm[1,1]}")
    print("="*80)

if __name__ == "__main__":
    try:
        validate()
    except Exception as e:
        print(f"❌ Erreur : {e}")
