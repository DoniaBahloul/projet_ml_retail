"""
preprocessing.py
================
Pipeline complet de préparation des données.

Étapes :
1. Parsing RegistDate (formats multiples UK/ISO/US)
2. Feature engineering depuis LastLoginIP
3. Feature engineering RFM / temporel
4. Suppression Newsletter (variance nulle)
5. Correction valeurs aberrantes (SupportTickets, Satisfaction)
6. Imputation Age (KNNImputer)
7. Encodage ordinal et One-Hot
8. Target encoding Country
9. Suppression multicolinéarité (|r| > 0.85)
10. Split Train/Test
11. StandardScaler (fit sur X_train uniquement — évite le data leakage)
12. Sauvegarde des données traitées

Usage:
    python src/preprocessing.py
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Chemins ───────────────────────────────────────────────────────────────────
DATA_RAW = Path("data/raw/retail_customers.csv")
DATA_PROCESSED_DIR = Path("data/processed")
TRAIN_TEST_DIR = Path("data/train_test")
MODELS_DIR = Path("models")

for d in [DATA_PROCESSED_DIR, TRAIN_TEST_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARSING — RegistDate
# ══════════════════════════════════════════════════════════════════════════════

def parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse RegistDate depuis des formats inconsistants (UK, ISO, US).
    Crée 4 features : RegYear, RegMonth, RegDay, RegWeekday.
    Supprime la colonne texte originale.
    """
    print("🗓️  Parsing RegistDate...")
    df["RegistDate"] = pd.to_datetime(
        df["RegistDate"],
        dayfirst=True,   # Priorité format UK : JJ/MM/AA
        errors="coerce", # NaT si format non reconnu
    )
    df["RegYear"] = df["RegistDate"].dt.year
    df["RegMonth"] = df["RegistDate"].dt.month
    df["RegDay"] = df["RegistDate"].dt.day
    df["RegWeekday"] = df["RegistDate"].dt.weekday

    # Imputer les NaT avec la médiane
    for col in ["RegYear", "RegMonth", "RegDay", "RegWeekday"]:
        df[col] = df[col].fillna(df[col].median()).astype(int)

    df.drop(columns=["RegistDate"], inplace=True)
    print(f"   → 4 features créées (RegYear, RegMonth, RegDay, RegWeekday)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — LastLoginIP
# ══════════════════════════════════════════════════════════════════════════════

def _is_private_ip(ip: str) -> int:
    """Retourne 1 si l'IP est privée (RFC 1918), 0 sinon."""
    try:
        parts = [int(x) for x in ip.strip().split(".")]
        if len(parts) != 4:
            return -1
        if parts[0] == 10:
            return 1
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return 1
        if parts[0] == 192 and parts[1] == 168:
            return 1
        return 0
    except Exception:
        return -1


def engineer_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des features depuis LastLoginIP :
    - IP_IsPrivate : 1 si IP privée (réseau interne / VPN), 0 sinon
    - IP_FirstOctet : premier octet (indication géographique approximative)
    Supprime la colonne LastLoginIP.
    """
    print("🌐 Feature engineering LastLoginIP...")
    df["IP_IsPrivate"] = df["LastLoginIP"].apply(_is_private_ip)
    df["IP_FirstOctet"] = df["LastLoginIP"].apply(
        lambda ip: int(ip.split(".")[0]) if isinstance(ip, str) and ip.count(".") == 3 else -1
    )
    df.drop(columns=["LastLoginIP"], inplace=True)
    print("   → 2 features créées (IP_IsPrivate, IP_FirstOctet)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING — RFM / comportemental
# ══════════════════════════════════════════════════════════════════════════════

def engineer_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features dérivées des métriques RFM et temporelles."""
    print("⚙️  Feature engineering RFM...")
    df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
    df["AvgBasketValue"] = df["MonetaryTotal"] / df["Frequency"].replace(0, 1)
    df["TenureRatio"] = df["Recency"] / (df["CustomerTenure"] + 1)
    df["FrequencyRatio"] = df["Frequency"] / (df["CustomerTenure"] + 1)
    print("   → 4 features créées")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUPPRESSION — Newsletter (variance nulle)
# ══════════════════════════════════════════════════════════════════════════════

def drop_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les features à variance nulle ou inutiles."""
    to_drop = []

    # Newsletter : toujours "Yes"
    if "Newsletter" in df.columns and df["Newsletter"].nunique() == 1:
        to_drop.append("Newsletter")

    # CustomerID : identifiant, pas une feature prédictive
    if "CustomerID" in df.columns:
        to_drop.append("CustomerID")

    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"🗑️  Features supprimées (variance nulle / identifiant) : {to_drop}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRECTION — Valeurs aberrantes SupportTickets & Satisfaction
# ══════════════════════════════════════════════════════════════════════════════

def fix_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les valeurs aberrantes connues :
    - SupportTickets : -1 et 999 → médiane des valeurs valides
    - Satisfaction   : -1 et 99  → médiane des valeurs valides
    """
    print("🔧 Correction des valeurs aberrantes...")

    # SupportTickets : valeurs valides entre 0 et 15
    valid_support = df.loc[df["SupportTickets"].between(0, 15), "SupportTickets"]
    median_support = valid_support.median()
    mask_bad_support = ~df["SupportTickets"].between(0, 15)
    n_fixed = mask_bad_support.sum()
    df.loc[mask_bad_support, "SupportTickets"] = median_support
    print(f"   SupportTickets : {n_fixed} valeurs aberrantes remplacées par médiane={median_support:.1f}")

    # Satisfaction : valeurs valides entre 1 et 5
    valid_sat = df.loc[df["Satisfaction"].between(1, 5), "Satisfaction"]
    median_sat = valid_sat.median()
    mask_bad_sat = ~df["Satisfaction"].between(1, 5)
    n_fixed_sat = mask_bad_sat.sum()
    df.loc[mask_bad_sat, "Satisfaction"] = median_sat
    print(f"   Satisfaction   : {n_fixed_sat} valeurs aberrantes remplacées par médiane={median_sat:.1f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. IMPUTATION — Age (30% manquants)
# ══════════════════════════════════════════════════════════════════════════════

def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de Age avec KNNImputer (k=5).
    Note : le fit se fera sur X_train uniquement lors du split.
    Pour cette étape de preprocessing global, on utilise la médiane.
    """
    print("🩹 Imputation Age (médiane)...")
    n_missing = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    print(f"   {n_missing} valeurs manquantes imputées par la médiane")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENCODAGE — Catégorielles
# ══════════════════════════════════════════════════════════════════════════════

# Mappings ordinaux (ordre logique)
ORDINAL_MAPS = {
    "AgeCategory": {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5, "Inconnu": -1},
    "SpendingCat": {"Low": 0, "Medium": 1, "High": 2, "VIP": 3},
    "LoyaltyLevel": {"Nouveau": 0, "Jeune": 1, "Établi": 2, "Ancien": 3, "Inconnu": -1},
    "ChurnRisk": {"Faible": 0, "Moyen": 1, "Élevé": 2, "Critique": 3},
    "BasketSize": {"Petit": 0, "Moyen": 1, "Grand": 2, "Inconnu": -1},
    "PreferredTime": {"Nuit": 0, "Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4},
    "RFMSegment": {"Dormants": 0, "Potentiels": 1, "Fidèles": 2, "Champions": 3},
}

# Features One-Hot
ONE_HOT_FEATURES = [
    "CustomerType", "FavoriteSeason", "Region",
    "WeekendPref", "ProdDiversity", "Gender", "AccountStatus",
]


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables catégorielles ordinales."""
    print("🔢 Encodage ordinal...")
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
            print(f"   {col} → encodé")
    return df


def encode_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables catégorielles nominales en One-Hot."""
    print("🔣 Encodage One-Hot...")
    cols_present = [c for c in ONE_HOT_FEATURES if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=False, dtype=int)
    print(f"   {len(cols_present)} features encodées → {df.shape[1]} colonnes total")
    return df


def encode_target_country(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """
    Target encoding pour Country :
    Remplace chaque pays par le taux de churn moyen observé dans le dataset.
    ⚠️ Dans un pipeline réel, ce calcul se fait sur X_train uniquement.
    """
    print("🌍 Target encoding Country...")
    if "Country" not in df.columns:
        return df
    country_churn_rate = df.groupby("Country")[target_col].mean()
    df["Country_TargetEnc"] = df["Country"].map(country_churn_rate)
    df["Country_TargetEnc"] = df["Country_TargetEnc"].fillna(df[target_col].mean())
    df.drop(columns=["Country"], inplace=True)
    print("   Country → Country_TargetEnc (taux churn moyen par pays)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUPPRESSION MULTICOLINÉARITÉ
# ══════════════════════════════════════════════════════════════════════════════

def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.85, target: str = TARGET) -> pd.DataFrame:
    """
    Supprime les features numériques fortement corrélées (|r| > threshold).
    Conserve la feature avec la plus forte corrélation avec la cible.
    """
    print(f"\n🔍 Suppression multicolinéarité (seuil |r| > {threshold})...")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    target_corr = df[num_cols].corrwith(df[target]).abs()

    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for corr_col in correlated:
            if corr_col not in to_drop and col not in to_drop:
                # Conserver celle la plus corrélée avec la target
                if target_corr.get(col, 0) >= target_corr.get(corr_col, 0):
                    to_drop.add(corr_col)
                else:
                    to_drop.add(col)

    if to_drop:
        df.drop(columns=list(to_drop), inplace=True)
        print(f"   {len(to_drop)} features supprimées : {sorted(to_drop)}")
    else:
        print("   Aucune paire fortement corrélée détectée.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 9. SPLIT TRAIN/TEST
# ══════════════════════════════════════════════════════════════════════════════

def split_and_save(df: pd.DataFrame, target: str = TARGET) -> tuple:
    """
    Sépare en X_train, X_test, y_train, y_test (80/20, stratifié).
    Sauvegarde les 4 fichiers CSV dans data/train_test/.
    """
    print(f"\n✂️  Split Train/Test (stratifié, test_size={TEST_SIZE})...")
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train.to_csv(TRAIN_TEST_DIR / "X_train.csv", index=False)
    X_test.to_csv(TRAIN_TEST_DIR / "X_test.csv", index=False)
    y_train.to_csv(TRAIN_TEST_DIR / "y_train.csv", index=False)
    y_test.to_csv(TRAIN_TEST_DIR / "y_test.csv", index=False)

    print(f"   X_train : {X_train.shape}, X_test : {X_test.shape}")
    print(f"   Churn train : {y_train.mean():.1%} | Churn test : {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 10. NORMALISATION (fit sur X_train uniquement — pas de data leakage)
# ══════════════════════════════════════════════════════════════════════════════

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Centre et réduit les features numériques.
    FIT uniquement sur X_train pour éviter le data leakage.
    TRANSFORM sur X_train et X_test.
    Sauvegarde le scaler dans models/.
    """
    print("\n📏 Normalisation (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print(f"   Scaler sauvegardé dans models/scaler.joblib")
    return X_train_scaled, X_test_scaled, scaler


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline() -> tuple:
    """
    Lance la chaîne complète de préparation des données.
    Retourne : (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    """
    print("=" * 60)
    print("🚀 DÉMARRAGE DU PIPELINE DE PREPROCESSING")
    print("=" * 60)

    # Chargement
    df = pd.read_csv(DATA_RAW)
    print(f"✅ Dataset chargé : {df.shape}")

    # Étape 1 — Parsing dates
    df = parse_registration_date(df)

    # Étape 2 — Feature engineering IP
    df = engineer_ip_features(df)

    # Étape 3 — Feature engineering RFM
    df = engineer_rfm_features(df)

    # Étape 4 — Suppression features inutiles
    df = drop_useless_features(df)

    # Étape 5 — Correction outliers
    df = fix_outliers(df)

    # Étape 6 — Imputation Age
    df = impute_age(df)

    # Étape 7a — Encodage ordinal
    df = encode_ordinal(df)

    # Étape 7b — Target encoding Country
    df = encode_target_country(df)

    # Étape 7c — One-Hot encoding
    df = encode_onehot(df)

    # Étape 8 — Suppression multicolinéarité
    df = remove_highly_correlated(df, threshold=0.85)

    # Sauvegarde données traitées (avant split)
    processed_path = DATA_PROCESSED_DIR / "retail_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"\n💾 Données traitées sauvegardées : {processed_path}")
    print(f"   Shape finale : {df.shape}")

    # Étape 9 — Split
    X_train, X_test, y_train, y_test = split_and_save(df)

    # Étape 10 — Normalisation
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Sauvegarder les noms de features
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING TERMINÉ")
    print(f"   Features finales : {len(feature_names)}")
    print("=" * 60)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


if __name__ == "__main__":
    run_preprocessing_pipeline()
