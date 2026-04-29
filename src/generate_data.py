"""
generate_data.py
================
Génère un dataset synthétique de clients e-commerce (cadeaux) avec 52 features.
Le dataset est intentionnellement imparfait pour l'apprentissage du preprocessing.

Usage:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
import warnings

warnings.filterwarnings("ignore")

# ── Reproductibilité ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 1000  # Nombre de clients


def generate_dataset(n: int = N) -> pd.DataFrame:
    """Génère le dataset complet avec problèmes de qualité intentionnels."""

    # ── 1. CustomerID ─────────────────────────────────────────────────────────
    customer_ids = np.random.choice(range(10000, 99999), size=n, replace=False)

    # ── Churn (cible) — 20% de churn ─────────────────────────────────────────
    churn = np.random.choice([0, 1], size=n, p=[0.80, 0.20])

    # ── 2-4. RFM de base ──────────────────────────────────────────────────────
    # Plages qui se CHEVAUCHENT pour créer de l'incertitude réaliste
    recency = np.where(
        churn == 1,
        np.random.randint(30, 401, size=n),   # churners: 30-400 (au lieu de 100-400)
        np.random.randint(0, 280, size=n),     # fidèles : 0-280 (au lieu de 0-180)
    )
    frequency = np.where(
        churn == 1,
        np.random.randint(1, 20, size=n),      # churners: 1-19 (au lieu de 1-8)
        np.random.randint(2, 51, size=n),       # fidèles : 2-50 (au lieu de 3-51)
    )
    monetary_total = np.where(
        churn == 1,
        np.random.uniform(50, 6000, size=n),   # churners: 50-6000 (au lieu de 50-1500)
        np.random.uniform(100, 15000, size=n), # fidèles : 100-15000 (au lieu de 300-15000)
    )
    # Quelques valeurs négatives (retours/remboursements)
    neg_mask = np.random.random(n) < 0.05
    monetary_total[neg_mask] = np.random.uniform(-5000, -1, size=neg_mask.sum())

    # MonetaryAvg : partiellement corrélé (70% signal + 30% bruit)
    signal_avg = monetary_total / frequency
    noise_avg = np.random.uniform(15, 450, size=n)
    monetary_avg = 0.7 * signal_avg + 0.3 * noise_avg + np.random.normal(0, 30, n)
    monetary_avg = np.clip(monetary_avg, 5, 500)

    monetary_std = np.abs(np.random.normal(50, 30, n))
    monetary_std = np.clip(monetary_std, 0, 500)

    # MonetaryMin/Max : corrélés avec bruit modéré
    monetary_min = monetary_total * np.random.uniform(0.05, 0.35, n) + np.random.normal(0, 150, n)
    monetary_min = np.clip(monetary_min, -5000, 5000)

    monetary_max = monetary_total * np.random.uniform(0.5, 1.1, n) + np.random.normal(0, 200, n)
    monetary_max = np.clip(monetary_max, 0, 10000)

    # ── 5. Quantités ──────────────────────────────────────────────────────────
    total_qty = (frequency * np.random.randint(2, 30, size=n)).astype(int)
    total_qty = np.clip(total_qty, -10000, 100000)

    avg_qty_per_trans = total_qty / frequency + np.random.normal(0, 2, n)
    avg_qty_per_trans = np.clip(avg_qty_per_trans, 1, 1000)

    min_quantity = np.random.randint(-8000, 0, size=n)
    max_quantity = np.random.randint(1, 8001, size=n)

    # ── 6. Temporel ───────────────────────────────────────────────────────────
    customer_tenure = np.random.randint(0, 731, size=n)
    first_purchase = np.random.randint(0, customer_tenure + 1)
    preferred_day = np.random.randint(0, 7, size=n)
    preferred_hour = np.random.randint(0, 24, size=n)
    preferred_month = np.random.randint(1, 13, size=n)
    weekend_ratio = np.random.uniform(0.0, 1.0, size=n)
    avg_days_between = np.random.uniform(0, 365, size=n)

    # ── 7. Produits ───────────────────────────────────────────────────────────
    unique_products = np.random.randint(1, 1001, size=n)
    unique_desc = np.clip(unique_products + np.random.randint(-10, 10, n), 1, 1000)
    avg_prod_per_trans = np.clip(
        unique_products / frequency + np.random.normal(0, 1, n), 1, 100
    )
    unique_countries = np.random.randint(1, 6, size=n)

    # ── 8. Anomalies transactionnelles ────────────────────────────────────────
    neg_qty_count = np.random.randint(0, 101, size=n)
    zero_price_count = np.random.randint(0, 51, size=n)
    cancelled_trans = np.random.randint(0, 51, size=n)
    return_ratio = np.random.uniform(0.0, 1.0, size=n)
    total_trans = np.random.randint(1, 10001, size=n)
    unique_invoices = np.clip(
        frequency + np.random.randint(0, 10, n), 1, 500
    ).astype(int)
    avg_lines_per_inv = np.clip(
        total_trans / unique_invoices + np.random.normal(0, 2, n), 1, 100
    )

    # ── 9. Age (30% manquants) ────────────────────────────────────────────────
    age = np.random.uniform(18, 81, size=n)
    missing_age_mask = np.random.random(n) < 0.30
    age = age.astype(object)
    age[missing_age_mask] = np.nan

    # ── 10. SupportTickets (valeurs aberrantes intentionnelles) ───────────────
    support_tickets = np.random.randint(0, 16, size=n).astype(float)
    # ~5% valeurs -1 (erreur système)
    support_tickets[np.random.random(n) < 0.05] = -1
    # ~3% valeurs 999 (overflow)
    support_tickets[np.random.random(n) < 0.03] = 999

    # ── 11. Satisfaction (valeurs aberrantes intentionnelles) ─────────────────
    satisfaction = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.1, 0.25, 0.35, 0.25]).astype(float)
    # ~5% valeurs -1 (non renseigné)
    satisfaction[np.random.random(n) < 0.05] = -1
    # ~3% valeurs 99 (code erreur)
    satisfaction[np.random.random(n) < 0.03] = 99

    # ── 12. Churn déjà défini ─────────────────────────────────────────────────

    # ── 13. Features catégorielles (avec bruit réaliste) ──────────────────────
    # 20% des churners gardent un segment "positif" (erreur réaliste)
    rfm_segments = np.where(
        churn == 1,
        np.random.choice(["Dormants", "Potentiels", "Fidèles", "Champions"],
                         size=n, p=[0.45, 0.35, 0.15, 0.05]),
        np.random.choice(["Champions", "Fidèles", "Potentiels", "Dormants"],
                         size=n, p=[0.3, 0.35, 0.20, 0.15]),
    )

    age_categories = []
    age_bins = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Inconnu"]
    for a in age:
        if pd.isna(a):
            age_categories.append("Inconnu")
        elif a < 25:
            age_categories.append("18-24")
        elif a < 35:
            age_categories.append("25-34")
        elif a < 45:
            age_categories.append("35-44")
        elif a < 55:
            age_categories.append("45-54")
        elif a < 65:
            age_categories.append("55-64")
        else:
            age_categories.append("65+")

    spending_cat = pd.cut(
        np.nan_to_num(monetary_total.astype(float), nan=0),
        bins=[-np.inf, 500, 2000, 6000, np.inf],
        labels=["Low", "Medium", "High", "VIP"],
    )

    customer_types = np.random.choice(
        ["Hyperactif", "Régulier", "Occasionnel", "Nouveau", "Perdu"],
        size=n,
        p=[0.1, 0.35, 0.25, 0.15, 0.15],
    )
    # Les churnés sont surtout "Perdu" ou "Occasionnel" mais pas toujours (bruit 20%)
    customer_types[churn == 1] = np.random.choice(
        ["Perdu", "Occasionnel", "Régulier", "Nouveau"],
        size=churn.sum(), p=[0.45, 0.30, 0.15, 0.10]
    )

    seasons = ["Hiver", "Printemps", "Été", "Automne"]
    favorite_season = np.random.choice(seasons, size=n)

    time_slots = ["Matin", "Midi", "Après-midi", "Soir", "Nuit"]
    preferred_time = np.random.choice(time_slots, size=n)

    regions = ["UK", "Europe_N", "Europe_S", "Europe_E", "Europe_C", "Asie", "Autre"]
    region = np.random.choice(regions, size=n, p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

    loyalty_levels = ["Nouveau", "Jeune", "Établi", "Ancien", "Inconnu"]
    loyalty_level = pd.cut(
        customer_tenure,
        bins=[-1, 30, 90, 180, 730, np.inf],
        labels=["Nouveau", "Jeune", "Établi", "Ancien", "Inconnu"],
    )

    churn_risk_options = ["Faible", "Moyen", "Élevé", "Critique"]
    # Bruit réaliste : certains churners évalués comme "Faible" risque et vice-versa
    churn_risk = np.where(
        churn == 1,
        np.random.choice(["Faible", "Moyen", "Élevé", "Critique"],
                         size=n, p=[0.10, 0.15, 0.40, 0.35]),
        np.random.choice(["Faible", "Moyen", "Élevé", "Critique"],
                         size=n, p=[0.45, 0.30, 0.18, 0.07]),
    )

    weekend_pref = np.random.choice(["Weekend", "Semaine", "Inconnu"], size=n, p=[0.3, 0.6, 0.1])

    basket_size = np.random.choice(["Petit", "Moyen", "Grand", "Inconnu"], size=n, p=[0.3, 0.4, 0.2, 0.1])

    prod_diversity = np.random.choice(["Spécialisé", "Modéré", "Explorateur"], size=n, p=[0.3, 0.4, 0.3])

    gender = np.random.choice(["M", "F", "Unknown"], size=n, p=[0.38, 0.55, 0.07])

    account_status_options = ["Active", "Suspended", "Pending", "Closed"]
    # Bruit réaliste : certains churners ont encore un compte "Active"
    account_status = np.where(
        churn == 1,
        np.random.choice(["Active", "Suspended", "Pending", "Closed"],
                         size=n, p=[0.15, 0.30, 0.10, 0.45]),
        np.random.choice(["Active", "Suspended", "Pending", "Closed"],
                         size=n, p=[0.80, 0.08, 0.07, 0.05]),
    )

    countries = [
        "United Kingdom", "France", "Germany", "Spain", "Netherlands",
        "Belgium", "Switzerland", "Australia", "Norway", "Sweden",
        "Denmark", "Finland", "Ireland", "USA", "Canada", "Japan",
        "Singapore", "UAE", "Portugal", "Italy",
    ]
    country_weights = [
        0.50, 0.07, 0.06, 0.04, 0.04,
        0.03, 0.03, 0.03, 0.02, 0.02,
        0.02, 0.01, 0.01, 0.02, 0.01, 0.01,
        0.01, 0.01, 0.02, 0.03,
    ]
    # Normalisation pour garantir la somme = 1.0
    total_w = sum(country_weights)
    country_weights = [w / total_w for w in country_weights]
    country = np.random.choice(countries, size=n, p=country_weights)

    # Newsletter : toujours "Yes" (feature inutile)
    newsletter = ["Yes"] * n

    # ── RegistDate — formats inconsistants intentionnels ──────────────────────
    def random_date():
        year = random.randint(2009, 2023)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        fmt = random.choice(["uk", "iso", "us"])
        if fmt == "uk":
            return f"{day:02d}/{month:02d}/{str(year)[2:]}"
        elif fmt == "iso":
            return f"{year}-{month:02d}-{day:02d}"
        else:
            return f"{month:02d}/{day:02d}/{year}"

    regist_date = [random_date() for _ in range(n)]

    # ── LastLoginIP ───────────────────────────────────────────────────────────
    def random_ip():
        private_chance = random.random()
        if private_chance < 0.3:
            return f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
        elif private_chance < 0.5:
            return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        else:
            return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    last_login_ip = [random_ip() for _ in range(n)]

    # ── Assemblage DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Recency": recency,
        "Frequency": frequency,
        "MonetaryTotal": monetary_total.round(2),
        "MonetaryAvg": monetary_avg.round(2),
        "MonetaryStd": monetary_std.round(2),
        "MonetaryMin": monetary_min.round(2),
        "MonetaryMax": monetary_max.round(2),
        "TotalQuantity": total_qty,
        "AvgQtyPerTrans": avg_qty_per_trans.round(2),
        "MinQuantity": min_quantity,
        "MaxQuantity": max_quantity,
        "CustomerTenure": customer_tenure,
        "FirstPurchase": first_purchase,
        "PreferredDay": preferred_day,
        "PreferredHour": preferred_hour,
        "PreferredMonth": preferred_month,
        "WeekendRatio": weekend_ratio.round(3),
        "AvgDaysBetween": avg_days_between.round(1),
        "UniqueProducts": unique_products,
        "UniqueDesc": unique_desc,
        "AvgProdPerTrans": avg_prod_per_trans.round(2),
        "UniqueCountries": unique_countries,
        "NegQtyCount": neg_qty_count,
        "ZeroPriceCount": zero_price_count,
        "CancelledTrans": cancelled_trans,
        "ReturnRatio": return_ratio.round(3),
        "TotalTrans": total_trans,
        "UniqueInvoices": unique_invoices,
        "AvgLinesPerInv": avg_lines_per_inv.round(2),
        "Age": age,
        "SupportTickets": support_tickets,
        "Satisfaction": satisfaction,
        "Churn": churn,
        # Catégorielles
        "RFMSegment": rfm_segments,
        "AgeCategory": age_categories,
        "SpendingCat": spending_cat.astype(str),
        "CustomerType": customer_types,
        "FavoriteSeason": favorite_season,
        "PreferredTime": preferred_time,
        "Region": region,
        "LoyaltyLevel": loyalty_level.astype(str),
        "ChurnRisk": churn_risk,
        "WeekendPref": weekend_pref,
        "BasketSize": basket_size,
        "ProdDiversity": prod_diversity,
        "Gender": gender,
        "AccountStatus": account_status,
        "Country": country,
        "Newsletter": newsletter,
        "RegistDate": regist_date,
        "LastLoginIP": last_login_ip,
    })

    return df


if __name__ == "__main__":
    output_path = Path("data/raw/retail_customers.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("🔄 Génération du dataset synthétique...")
    df = generate_dataset(N)

    df.to_csv(output_path, index=False)
    print(f"✅ Dataset généré : {output_path}")
    print(f"   Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Taux de churn : {df['Churn'].mean():.1%}")
    print(f"   NaN dans Age : {df['Age'].isna().sum()} ({df['Age'].isna().mean():.1%})")
    print(f"   Colonnes : {list(df.columns)}")
