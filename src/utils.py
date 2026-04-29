"""
utils.py
========
Fonctions utilitaires réutilisables pour l'ensemble du pipeline ML.

Contient :
- Chargement / sauvegarde de données et modèles
- Visualisations (corrélation, PCA, distributions, boxplots)
- Métriques d'évaluation (classification, régression)
- Détection d'outliers (IQR)
"""

import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ── Palette & style Seaborn ───────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
PALETTE = "coolwarm"

# ── Chemins par défaut ────────────────────────────────────────────────────────
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_TRAIN_TEST = Path("data/train_test")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

for _dir in [DATA_RAW, DATA_PROCESSED, DATA_TRAIN_TEST, MODELS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & SAUVEGARDE
# ═════════════════════════════════════════════════════════════════════════════

def load_data(path: str | Path = DATA_RAW / "retail_customers.csv") -> pd.DataFrame:
    """Charge un CSV et affiche un résumé rapide."""
    df = pd.read_csv(path)
    print(f"✅ Données chargées depuis {path}")
    print(f"   Shape : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   NaN total : {df.isna().sum().sum()}")
    return df


def save_model(model, filename: str, directory: Path = MODELS_DIR) -> Path:
    """Sauvegarde un modèle entraîné avec joblib."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    joblib.dump(model, path)
    print(f"💾 Modèle sauvegardé : {path}")
    return path


def load_model(filename: str, directory: Path = MODELS_DIR):
    """Charge un modèle sauvegardé avec joblib."""
    path = directory / filename
    model = joblib.load(path)
    print(f"📂 Modèle chargé : {path}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 2. ANALYSE EXPLORATOIRE
# ═════════════════════════════════════════════════════════════════════════════

def describe_dataframe(df: pd.DataFrame) -> None:
    """Affiche un résumé complet du DataFrame."""
    print("=" * 60)
    print(f"SHAPE     : {df.shape}")
    print(f"MÉMOIRE   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("-" * 60)
    print("TYPES DE COLONNES :")
    print(df.dtypes.value_counts())
    print("-" * 60)
    print("VALEURS MANQUANTES (top 10) :")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100).round(1)
    print(pd.concat([miss, miss_pct], axis=1, keys=["NaN", "%"])[miss > 0].head(10))
    print("=" * 60)


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Détecte les outliers par la méthode IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
    Retourne un DataFrame résumé.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        results.append({
            "Feature": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower": round(lower, 2),
            "Upper": round(upper, 2),
            "N_Outliers": int(outlier_mask.sum()),
            "%_Outliers": round(outlier_mask.mean() * 100, 1),
        })
    return pd.DataFrame(results).sort_values("%_Outliers", ascending=False)


# ═════════════════════════════════════════════════════════════════════════════
# 3. VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

def plot_missing_values(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Heatmap des valeurs manquantes."""
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print("✅ Aucune valeur manquante dans le dataset.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    missing.plot(kind="barh", ax=ax, color="coral", edgecolor="white")
    ax.set_title("Taux de valeurs manquantes par feature", fontsize=14, fontweight="bold")
    ax.set_xlabel("Proportion manquante")
    ax.axvline(0.3, color="red", linestyle="--", alpha=0.7, label="Seuil 30%")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    threshold: float = 0.0,
    figsize: tuple = (16, 14),
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Heatmap de corrélation sur les features numériques.
    threshold : afficher uniquement les corrélations >= |threshold|.
    Retourne la matrice de corrélation.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=PALETTE,
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.3,
        linecolor="grey",
        square=True,
        ax=ax,
        annot_kws={"size": 6},
    )
    ax.set_title("Matrice de corrélation — Features numériques", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

    # Afficher les paires fortement corrélées
    if threshold > 0:
        corr_pairs = (
            corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]
        corr_pairs = corr_pairs[corr_pairs["Correlation"].abs() >= threshold]
        print(f"\n🔴 Paires fortement corrélées (|r| ≥ {threshold}) :")
        print(corr_pairs.sort_values("Correlation", ascending=False).to_string(index=False))

    return corr


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    n_components_selected: int | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Courbe de variance cumulée expliquée par l'ACP.
    explained_variance_ratio : sortie de pca.explained_variance_ratio_
    """
    cumulative = np.cumsum(explained_variance_ratio) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Variance par composante
    axes[0].bar(
        range(1, len(explained_variance_ratio) + 1),
        explained_variance_ratio * 100,
        color="steelblue",
        edgecolor="white",
    )
    axes[0].set_xlabel("Composante principale")
    axes[0].set_ylabel("Variance expliquée (%)")
    axes[0].set_title("Variance par composante")

    # Variance cumulée
    axes[1].plot(range(1, len(cumulative) + 1), cumulative, marker="o", color="darkorange")
    axes[1].axhline(80, color="red", linestyle="--", alpha=0.7, label="80%")
    axes[1].axhline(95, color="green", linestyle="--", alpha=0.7, label="95%")
    if n_components_selected:
        axes[1].axvline(
            n_components_selected, color="purple", linestyle="--",
            label=f"Sélectionnées : {n_components_selected}",
        )
    axes[1].set_xlabel("Nombre de composantes")
    axes[1].set_ylabel("Variance cumulée (%)")
    axes[1].set_title("Variance cumulée expliquée (ACP)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_distributions(df: pd.DataFrame, columns: list[str], save_path: Path | None = None) -> None:
    """Histogrammes + KDE pour une liste de features numériques."""
    n = len(columns)
    cols_per_row = 4
    n_rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 4, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in df.columns:
            df[col].dropna().plot(kind="hist", ax=axes[i], bins=30, color="steelblue",
                                  edgecolor="white", alpha=0.8, density=True)
            df[col].dropna().plot(kind="kde", ax=axes[i], color="darkorange")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel("")

    # Masquer les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribution des features numériques", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_boxplots(df: pd.DataFrame, columns: list[str], save_path: Path | None = None) -> None:
    """Boxplots pour détecter les outliers visuellement."""
    n = len(columns)
    cols_per_row = 4
    n_rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 4, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in df.columns:
            df[[col]].dropna().boxplot(ax=axes[i], vert=True, patch_artist=True,
                                        boxprops=dict(facecolor="steelblue", alpha=0.7))
            axes[i].set_title(col, fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots — Détection des outliers", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_pca_2d(
    X_pca: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Projection ACP 2D",
    save_path: Path | None = None,
) -> None:
    """Scatter plot de la projection ACP en 2 dimensions."""
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab10",
        alpha=0.6,
        edgecolors="white",
        linewidth=0.3,
        s=40,
    )
    if labels is not None:
        plt.colorbar(scatter, ax=ax, label="Cluster / Classe")
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 4. ÉVALUATION DES MODÈLES
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Modèle",
    save_path: Path | None = None,
) -> dict:
    """
    Évalue un modèle de classification et affiche :
    - Rapport de classification
    - Matrice de confusion
    - Courbe ROC
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n{'=' * 60}")
    print(f"📊 ÉVALUATION — {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if auc:
        print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Figure : Confusion Matrix + ROC
    n_plots = 2 if y_proba is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Fidèle", "Churné"], yticklabels=["Fidèle", "Churné"])
    axes[0].set_title(f"Matrice de confusion — {model_name}")
    axes[0].set_xlabel("Prédit")
    axes[0].set_ylabel("Réel")

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
        axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("Courbe ROC")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def evaluate_regressor(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Modèle",
    save_path: Path | None = None,
) -> dict:
    """
    Évalue un modèle de régression et affiche :
    - MAE, MSE, RMSE, R²
    - Graphique prédictions vs réelles
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"📊 ÉVALUATION RÉGRESSION — {model_name}")
    print(f"{'=' * 60}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Prédit vs réel
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="steelblue", edgecolors="white", linewidth=0.3)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Parfait")
    axes[0].set_xlabel("Valeurs réelles")
    axes[0].set_ylabel("Valeurs prédites")
    axes[0].set_title(f"Prédit vs Réel — {model_name}")
    axes[0].legend()

    # Résidus
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.4, color="coral", edgecolors="white", linewidth=0.3)
    axes[1].axhline(0, color="navy", linestyle="--", lw=2)
    axes[1].set_xlabel("Valeurs prédites")
    axes[1].set_ylabel("Résidus")
    axes[1].set_title("Distribution des résidus")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Importance des features",
    save_path: Path | None = None,
) -> None:
    """Barplot horizontal des top_n features les plus importantes."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
