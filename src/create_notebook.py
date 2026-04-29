"""Script pour créer le notebook d'exploration."""
import json
from pathlib import Path

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = [
    md("# Atelier ML Retail - Exploration et Preparation des Donnees\n\n*GI2 | 2025-2026*"),
    md("## 0. Imports et Configuration"),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from src.utils import (\n"
        "    describe_dataframe, detect_outliers_iqr,\n"
        "    plot_missing_values, plot_correlation_heatmap,\n"
        "    plot_pca_variance, plot_distributions, plot_boxplots, plot_pca_2d\n"
        ")\n"
        "sns.set_theme(style='darkgrid', palette='muted')\n"
        "plt.rcParams['figure.dpi'] = 100\n"
        "print('Imports OK')"
    ),
    md("## 1. Chargement des Donnees"),
    code(
        "df = pd.read_csv('../data/raw/retail_customers.csv')\n"
        "print(f'Shape : {df.shape}')\n"
        "df.head()"
    ),
    code("describe_dataframe(df)"),
    md("## 2. Valeurs Manquantes"),
    code(
        "plot_missing_values(df)\n"
        "missing = df.isna().sum().sort_values(ascending=False)\n"
        "missing[missing > 0]"
    ),
    md("## 3. Distributions des Features Numeriques"),
    code(
        "num_cols = [\n"
        "    'Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg',\n"
        "    'Age', 'CustomerTenure', 'SupportTickets', 'Satisfaction',\n"
        "    'WeekendRatio', 'ReturnRatio', 'UniqueProducts', 'CancelledTrans'\n"
        "]\n"
        "plot_distributions(df, num_cols)"
    ),
    md("## 4. Detection des Outliers (IQR)"),
    code(
        "outliers = detect_outliers_iqr(\n"
        "    df, ['SupportTickets', 'Satisfaction', 'MonetaryTotal', 'TotalQuantity']\n"
        ")\n"
        "print(outliers.to_string())\n"
        "plot_boxplots(df, ['SupportTickets', 'Satisfaction', 'MonetaryTotal', 'ReturnRatio'])"
    ),
    md("## 5. Analyse du Churn (Variable Cible)"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "\n"
        "# Distribution Churn\n"
        "vals = df['Churn'].value_counts()\n"
        "axes[0].bar(['Fidele (0)', 'Churn (1)'], vals.values, color=['steelblue', 'coral'], edgecolor='white')\n"
        "axes[0].set_title('Distribution du Churn')\n"
        "for i, v in enumerate(vals.values):\n"
        "    pct = v / len(df)\n"
        "    axes[0].text(i, v + 5, f'{v} ({pct:.1%})', ha='center')\n"
        "\n"
        "# Churn par segment RFM\n"
        "churn_seg = df.groupby('RFMSegment')['Churn'].mean().sort_values()\n"
        "churn_seg.plot(kind='barh', ax=axes[1], color='darkorange', edgecolor='white')\n"
        "axes[1].set_title('Taux de churn par segment RFM')\n"
        "axes[1].set_xlabel('Taux de churn')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 6. Correlations entre Features"),
    code(
        "num_df = df.select_dtypes(include=['number']).drop(columns=['CustomerID'], errors='ignore')\n"
        "# Heatmap correlation\n"
        "corr = plot_correlation_heatmap(num_df, threshold=0.85)"
    ),
    code(
        "# Paires fortement correlees avec Churn\n"
        "churn_corr = num_df.corrwith(num_df['Churn']).abs().sort_values(ascending=False)\n"
        "print('Top 15 features correlees avec Churn :')\n"
        "print(churn_corr.head(15))"
    ),
    md(
        "## 7. Analyse en Composantes Principales (ACP)\n\n"
        "Objectif : reduire les 52 features en 2-10 composantes tout en conservant la variance."
    ),
    code(
        "num_clean = num_df.fillna(num_df.median())\n"
        "X_scaled = StandardScaler().fit_transform(\n"
        "    num_clean.drop(columns=['Churn'], errors='ignore')\n"
        ")\n"
        "\n"
        "pca_full = PCA(random_state=42)\n"
        "pca_full.fit(X_scaled)\n"
        "\n"
        "plot_pca_variance(pca_full.explained_variance_ratio_)\n"
        "\n"
        "cumvar = np.cumsum(pca_full.explained_variance_ratio_)\n"
        "n80 = int(np.searchsorted(cumvar, 0.80)) + 1\n"
        "n95 = int(np.searchsorted(cumvar, 0.95)) + 1\n"
        "print(f'Composantes pour 80% de variance : {n80}')\n"
        "print(f'Composantes pour 95% de variance : {n95}')"
    ),
    code(
        "# Projection 2D coloree par Churn\n"
        "pca2 = PCA(n_components=2, random_state=42)\n"
        "X_pca2 = pca2.fit_transform(X_scaled)\n"
        "labels = num_clean['Churn'].values if 'Churn' in num_clean.columns else None\n"
        "plot_pca_2d(X_pca2, labels=labels, title='Projection ACP 2D - colore par Churn')"
    ),
    md("## 8. Feature Engineering"),
    code(
        "df_fe = df.copy()\n"
        "\n"
        "# 1. Ratio depenses / recency\n"
        "df_fe['MonetaryPerDay'] = df_fe['MonetaryTotal'] / (df_fe['Recency'] + 1)\n"
        "\n"
        "# 2. Panier moyen recalcule\n"
        "df_fe['AvgBasketValue'] = df_fe['MonetaryTotal'] / df_fe['Frequency'].replace(0, 1)\n"
        "\n"
        "# 3. Anciennete vs activite recente\n"
        "df_fe['TenureRatio'] = df_fe['Recency'] / (df_fe['CustomerTenure'] + 1)\n"
        "\n"
        "# 4. Frequence relative\n"
        "df_fe['FrequencyRatio'] = df_fe['Frequency'] / (df_fe['CustomerTenure'] + 1)\n"
        "\n"
        "# 5. Extraction depuis RegistDate\n"
        "df_fe['RegistDate'] = pd.to_datetime(df_fe['RegistDate'], dayfirst=True, errors='coerce')\n"
        "df_fe['RegYear'] = df_fe['RegistDate'].dt.year\n"
        "df_fe['RegMonth'] = df_fe['RegistDate'].dt.month\n"
        "\n"
        "print('Nouvelles features creees :')\n"
        "print(df_fe[['MonetaryPerDay', 'AvgBasketValue', 'TenureRatio', 'FrequencyRatio', 'RegYear', 'RegMonth']].describe().round(2))"
    ),
    md("## 9. Lancer le Pipeline de Preprocessing Complet"),
    code(
        "import subprocess, sys\n"
        "res = subprocess.run(\n"
        "    [sys.executable, '../src/preprocessing.py'],\n"
        "    capture_output=True, text=True, cwd='..'\n"
        ")\n"
        "print(res.stdout)\n"
        "if res.returncode != 0:\n"
        "    print('ERREUR:', res.stderr)"
    ),
    code(
        "# Verifier les donnees produites\n"
        "df_proc = pd.read_csv('../data/processed/retail_processed.csv')\n"
        "X_train = pd.read_csv('../data/train_test/X_train.csv')\n"
        "X_test = pd.read_csv('../data/train_test/X_test.csv')\n"
        "y_train = pd.read_csv('../data/train_test/y_train.csv').squeeze()\n"
        "y_test = pd.read_csv('../data/train_test/y_test.csv').squeeze()\n"
        "\n"
        "print(f'Shape apres preprocessing : {df_proc.shape}')\n"
        "print(f'X_train : {X_train.shape} | X_test : {X_test.shape}')\n"
        "y_mean_train = y_train.mean()\n"
        "y_mean_test = y_test.mean()\n"
        "print(f'Churn train : {y_mean_train:.1%} | Churn test : {y_mean_test:.1%}')"
    ),
    md(
        "## Resume des Traitements\n\n"
        "| Etape | Action | Resultat |\n"
        "|---|---|---|\n"
        "| Valeurs manquantes | Imputation Age (mediane) | 267 NaN corriges |\n"
        "| Outliers | Correction SupportTickets (-1, 999) | 77 corriges |\n"
        "| Outliers | Correction Satisfaction (-1, 99) | 76 corriges |\n"
        "| Formats | Parsing RegistDate multi-format | 4 features creees |\n"
        "| Variance nulle | Suppression Newsletter | 1 feature supprimee |\n"
        "| Feature brute | Engineering LastLoginIP | 2 features creees |\n"
        "| Multicolinearite | Seuil |r| > 0.85 | ~5 features supprimees |\n"
        "| **Total** | **52 features brutes** | **74 features finales** |\n"
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path("notebooks/exploration.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook cree : {out} ({len(cells)} cellules)")
