# 🛍️ Atelier ML Retail — Analyse Comportementale Clientèle E-commerce

**Module Machine Learning — GI2 | Année Universitaire 2025-2026**  
Préparé par : *Fadoua Drira*

> Chaîne complète de traitement Data Science : Exploration → Préparation → Modélisation → Évaluation → Déploiement

---

## 📋 Description du Projet

Ce projet implémente une **analyse comportementale de la clientèle** d'un e-commerce de cadeaux.  
À partir d'un dataset de **1 000 clients** avec **52 features** (intentionnellement imparfait), on réalise :

- 🔍 **Exploration** des données (valeurs manquantes, outliers, distributions)
- 🧹 **Préparation** complète du pipeline ML (encoding, normalisation, feature engineering)
- 🤖 **Modélisation** : Clustering KMeans, Classification Churn (Random Forest), Régression MonetaryTotal
- 🚀 **Déploiement** via une interface web Flask

---

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone <url_du_repo>
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel

```bash
# Création
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (Linux/Mac)
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 📁 Structure du Projet

```
projet_ml_retail/
├── data/
│   ├── raw/                    # Données brutes (retail_customers.csv)
│   ├── processed/              # Données nettoyées (retail_processed.csv)
│   └── train_test/             # Split X_train, X_test, y_train, y_test
├── notebooks/
│   └── exploration.ipynb       # EDA complet (Jupyter)
├── src/
│   ├── generate_data.py        # Génération du dataset synthétique
│   ├── utils.py                # Fonctions utilitaires (plots, métriques)
│   ├── preprocessing.py        # Pipeline de préparation complet
│   ├── train_model.py          # Entraînement des 3 modèles ML
│   └── predict.py              # Prédiction sur nouveaux clients
├── models/
│   ├── scaler.joblib           # StandardScaler entraîné
│   ├── kmeans_model.joblib     # Modèle KMeans
│   ├── churn_rf_model.joblib   # Random Forest Churn
│   ├── monetary_regressor.joblib  # Ridge/Lasso MonetaryTotal
│   └── feature_names.joblib   # Liste des features
├── app/
│   ├── app.py                  # Application Flask
│   ├── templates/              # Pages HTML (index, result, dashboard)
│   └── static/                 # CSS
├── reports/                    # Graphiques générés
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Guide d'Utilisation

### Étape 1 — Générer les données

```bash
python src/generate_data.py
```
→ Crée `data/raw/retail_customers.csv` (1 000 clients, 52 features)

### Étape 2 — Preprocessing

```bash
python src/preprocessing.py
```
→ Nettoie, encode et normalise les données  
→ Produit `data/processed/retail_processed.csv` et les fichiers train/test  
→ Sauvegarde le `scaler.joblib` dans `models/`

### Étape 3 — Entraîner les modèles

```bash
python src/train_model.py
```
→ Entraîne et sauvegarde 3 modèles dans `models/`  
→ Génère les graphiques d'évaluation dans `reports/`

### Étape 4 — Prédiction (exemple)

```bash
python src/predict.py
```
→ Tente une prédiction sur 2 clients exemples (fidèle vs risqué)

### Étape 5 — Lancer l'interface Flask

```bash
python app/app.py
```
→ Ouvrir le navigateur : [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Exploration interactive (Jupyter)

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 🤖 Modèles ML

| Modèle | Tâche | Algorithme | Métrique |
|--------|-------|-----------|---------|
| Clustering | Segmentation clients | KMeans | Silhouette Score |
| Classification | Prédiction Churn | Random Forest + GridSearchCV | AUC-ROC, F1 |
| Régression | Prédiction MonetaryTotal | Ridge / Lasso | R², RMSE |

---

## ⚠️ Points pédagogiques importants

- **Pas de data leakage** : `StandardScaler` et `KNNImputer` fittés **uniquement sur `X_train`**
- La variable **Churn** (target) n'est **jamais normalisée**
- Le déséquilibre de classes (~20% churn) est géré avec `class_weight='balanced'` et `stratify=y`
- La multicolinéarité est traitée (seuil |r| > 0.85)

---

## 📊 Problèmes de qualité traités

| Problème | Feature | Traitement |
|---------|---------|-----------|
| 30% NaN | `Age` | Imputation médiane |
| Valeurs aberrantes | `SupportTickets` (-1, 999) | Remplacement par médiane |
| Valeurs aberrantes | `Satisfaction` (-1, 99) | Remplacement par médiane |
| Formats inconsistants | `RegistDate` | Parsing multi-format |
| Variance nulle | `Newsletter` (= "Yes") | Suppression |
| Feature brute | `LastLoginIP` | Feature engineering (IP privée/publique) |
| Multicolinéarité | Monétaires croisés | Suppression seuil |r| > 0.85 |

---

*Atelier Pratique — E-commerce de Cadeaux | GI2 2025-2026*