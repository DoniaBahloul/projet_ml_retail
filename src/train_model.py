"""
train_model.py
==============
Entraîne 3 types de modèles ML sur le dataset retail :

1. Clustering KMeans   → segmentation clients (non supervisé)
2. Classification RF   → prédiction Churn (supervisé)
3. Régression Ridge    → prédiction MonetaryTotal (supervisé)

Usage:
    python src/train_model.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif — pas de fenêtre GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Chemins ───────────────────────────────────────────────────────────────────
TRAIN_TEST_DIR = Path("data/train_test")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_CLASS = "Churn"
TARGET_REG = "MonetaryTotal"


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def load_train_test():
    """Charge les données splittées depuis data/train_test/."""
    X_train = pd.read_csv(TRAIN_TEST_DIR / "X_train.csv")
    X_test = pd.read_csv(TRAIN_TEST_DIR / "X_test.csv")
    y_train = pd.read_csv(TRAIN_TEST_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(TRAIN_TEST_DIR / "y_test.csv").squeeze()
    print(f"✅ Données chargées")
    print(f"   X_train : {X_train.shape} | X_test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_for_training(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Charge le scaler déjà entraîné et transforme les données."""
    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print("📏 Scaler chargé depuis models/scaler.joblib")
        return scaler.transform(X_train), scaler.transform(X_test)
    else:
        # Fallback : fit sur X_train si le scaler n'existe pas
        print("⚠️  Scaler non trouvé — fit sur X_train (attention: run preprocessing d'abord)")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        joblib.dump(scaler, scaler_path)
        return X_train_s, X_test_s


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — CLUSTERING KMEANS
# ══════════════════════════════════════════════════════════════════════════════

def train_clustering(X_train_scaled: np.ndarray, feature_names: list[str]) -> KMeans:
    """
    Clustering KMeans avec sélection du nombre optimal de clusters
    par la méthode Elbow + Silhouette.
    """
    print("\n" + "=" * 60)
    print("🔵 MODULE 1 — CLUSTERING KMEANS")
    print("=" * 60)

    # ── Méthode Elbow ─────────────────────────────────────────────────────────
    k_range = range(2, 11)
    inertias = []
    silhouettes = []

    print("   Recherche du k optimal (Elbow + Silhouette)...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_train_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_train_scaled, labels, sample_size=500))

    # Visualisation Elbow
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
    axes[0].set_xlabel("Nombre de clusters k")
    axes[0].set_ylabel("Inertie")
    axes[0].set_title("Méthode Elbow")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, marker="s", color="darkorange")
    axes[1].set_xlabel("Nombre de clusters k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Score Silhouette")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Sélection du nombre de clusters — KMeans", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "clustering_elbow.png", dpi=120, bbox_inches="tight")
    plt.close()

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"   → k optimal (Silhouette max) : {best_k}")

    # ── Entraînement final ────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_train_scaled)

    sil = silhouette_score(X_train_scaled, kmeans.labels_, sample_size=500)
    print(f"   Inertie finale   : {kmeans.inertia_:.0f}")
    print(f"   Silhouette Score : {sil:.4f}")

    # Distribution des clusters
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    print(f"   Distribution : {dict(zip(unique, counts))}")

    # ── ACP 2D pour visualisation ─────────────────────────────────────────────
    pca_viz = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca_viz.fit_transform(X_train_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=kmeans.labels_, cmap="tab10",
        alpha=0.6, edgecolors="white", linewidth=0.3, s=40,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel(f"PC1 ({pca_viz.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca_viz.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"Segmentation clients — KMeans (k={best_k})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "clustering_pca2d.png", dpi=120, bbox_inches="tight")
    plt.close()

    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.joblib")
    print("💾 Modèle sauvegardé : models/kmeans_model.joblib")

    return kmeans


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — CLASSIFICATION CHURN (Random Forest + GridSearch)
# ══════════════════════════════════════════════════════════════════════════════

def train_churn_classifier(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> RandomForestClassifier:
    """
    Entraîne un Random Forest pour prédire le Churn.
    Utilise GridSearchCV pour l'optimisation des hyperparamètres.
    Gère le déséquilibre de classes avec class_weight='balanced'.
    """
    print("\n" + "=" * 60)
    print("🟠 MODULE 2 — CLASSIFICATION CHURN (Random Forest)")
    print("=" * 60)
    print(f"   Déséquilibre classes : {y_train.mean():.1%} churned")

    # ── GridSearchCV ──────────────────────────────────────────────────────────
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "class_weight": ["balanced"],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    print("\n   🔍 GridSearchCV en cours...")
    grid_search.fit(X_train_scaled, y_train)

    best_rf = grid_search.best_estimator_
    print(f"\n   Meilleurs hyperparamètres : {grid_search.best_params_}")
    print(f"   Meilleur F1 (CV)          : {grid_search.best_score_:.4f}")

    # ── Évaluation sur le test set ────────────────────────────────────────────
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, roc_auc_score, roc_curve,
    )

    y_pred = best_rf.predict(X_test_scaled)
    y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n   📈 Résultats sur X_test :")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # ── Figures : Confusion Matrix + ROC ─────────────────────────────────────
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Fidèle", "Churné"], yticklabels=["Fidèle", "Churné"])
    axes[0].set_title("Matrice de confusion — Churn")
    axes[0].set_xlabel("Prédit")
    axes[0].set_ylabel("Réel")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "navy", lw=1.5, linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Courbe ROC — Random Forest Churn")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_rf_evaluation.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── Feature Importance ────────────────────────────────────────────────────
    importances = best_rf.feature_importances_
    top_n = 20
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="steelblue", edgecolor="white",
    )
    ax.set_title(f"Top {top_n} features — Random Forest Churn", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    joblib.dump(best_rf, MODELS_DIR / "churn_rf_model.joblib")
    print("💾 Modèle sauvegardé : models/churn_rf_model.joblib")

    return best_rf


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — RÉGRESSION MonetaryTotal (Ridge + Lasso)
# ══════════════════════════════════════════════════════════════════════════════

def train_monetary_regressor(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train_full: pd.Series,
    y_test_full: pd.Series,
    feature_names: list[str],
) -> Ridge:
    """
    Entraîne des modèles de régression Ridge et Lasso pour prédire
    MonetaryTotal depuis le dataset complet (hors Churn).
    Compare les deux modèles.
    """
    print("\n" + "=" * 60)
    print("🟣 MODULE 3 — RÉGRESSION MonetaryTotal (Ridge vs Lasso)")
    print("=" * 60)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=5000),
    }

    results = {}
    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        # Validation croisée
        cv_scores = cross_val_score(model, X_train_scaled, y_train_full, cv=5, scoring="r2")
        print(f"\n   {name} — R² CV (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Entraînement final
        model.fit(X_train_scaled, y_train_full)
        y_pred = model.predict(X_test_scaled)

        mae_ = mean_absolute_error(y_test_full, y_pred)
        rmse_ = np.sqrt(mean_squared_error(y_test_full, y_pred))
        r2_ = r2_score(y_test_full, y_pred)

        print(f"   {name} — MAE={mae_:.1f} | RMSE={rmse_:.1f} | R²={r2_:.4f}")
        results[name] = {"mae": mae_, "rmse": rmse_, "r2": r2_, "y_pred": y_pred, "model": model}

        if r2_ > best_r2:
            best_r2 = r2_
            best_model = model
            best_name = name

    print(f"\n   🏆 Meilleur modèle : {best_name} (R²={best_r2:.4f})")

    # ── Visualisation Ridge (prédit vs réel + résidus) ────────────────────────
    best_pred = results[best_name]["y_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test_full, best_pred, alpha=0.4, color="steelblue", edgecolors="white", linewidth=0.3)
    mn, mx = min(y_test_full.min(), best_pred.min()), max(y_test_full.max(), best_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Parfait")
    axes[0].set_xlabel("Valeurs réelles (£)")
    axes[0].set_ylabel("Valeurs prédites (£)")
    axes[0].set_title(f"Prédit vs Réel — {best_name}")
    axes[0].legend()

    residuals = y_test_full - best_pred
    axes[1].scatter(best_pred, residuals, alpha=0.4, color="coral", edgecolors="white", linewidth=0.3)
    axes[1].axhline(0, color="navy", linestyle="--", lw=2)
    axes[1].set_xlabel("Valeurs prédites")
    axes[1].set_ylabel("Résidus")
    axes[1].set_title(f"Résidus — {best_name}")

    plt.suptitle(f"Régression MonetaryTotal — {best_name} (R²={best_r2:.4f})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "regression_monetary_evaluation.png", dpi=120, bbox_inches="tight")
    plt.close()

    joblib.dump(best_model, MODELS_DIR / "monetary_regressor.joblib")
    print(f"💾 Modèle sauvegardé : models/monetary_regressor.joblib")

    return best_model


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_training_pipeline():
    """Lance les 3 modules de modélisation à la suite."""
    print("=" * 60)
    print("🚀 DÉMARRAGE DU PIPELINE DE MODÉLISATION")
    print("=" * 60)

    # Chargement
    X_train_df, X_test_df, y_train, y_test = load_train_test()

    # Charger feature names
    feature_names_path = MODELS_DIR / "feature_names.joblib"
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
    else:
        feature_names = list(X_train_df.columns)

    # Normalisation
    X_train_s, X_test_s = scale_for_training(X_train_df, X_test_df)

    # ── Préparer target régression CORRECTEMENT ────────────────────────────────
    # Extraire MonetaryTotal directement depuis X_train/X_test (même lignes garanties)
    if "MonetaryTotal" in X_train_df.columns:
        y_monetary_train = X_train_df["MonetaryTotal"].copy()
        y_monetary_test = X_test_df["MonetaryTotal"].copy()

        # Retirer MonetaryTotal des features (on ne peut pas prédire X à partir de X)
        # Garder MonetaryAvg, MonetaryStd comme signaux partiels
        reg_drop_cols = ["MonetaryTotal"]
        X_train_reg = X_train_df.drop(columns=reg_drop_cols, errors="ignore")
        X_test_reg = X_test_df.drop(columns=reg_drop_cols, errors="ignore")

        # Scaler dédié à la régression
        scaler_reg = StandardScaler()
        X_train_reg_s = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_s = scaler_reg.transform(X_test_reg)
        reg_feature_names = list(X_train_reg.columns)
        print(f"📊 Régression : {len(reg_feature_names)} features (MonetaryTotal retiré)")
    else:
        X_train_reg_s = X_train_s
        X_test_reg_s = X_test_s
        y_monetary_train = y_train
        y_monetary_test = y_test
        reg_feature_names = feature_names

    # ── Module 1 : Clustering ─────────────────────────────────────────────────
    kmeans = train_clustering(X_train_s, feature_names)

    # ── Module 2 : Classification Churn ──────────────────────────────────────
    rf_churn = train_churn_classifier(X_train_s, X_test_s, y_train, y_test, feature_names)

    # ── Module 3 : Régression Monetary ───────────────────────────────────────
    regressor = train_monetary_regressor(
        X_train_reg_s, X_test_reg_s,
        y_monetary_train.reset_index(drop=True),
        y_monetary_test.reset_index(drop=True),
        reg_feature_names,
    )

    print("\n" + "=" * 60)
    print("✅ MODÉLISATION TERMINÉE")
    print("   Modèles sauvegardés dans models/")
    print("   Rapports sauvegardés dans reports/")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()
