import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.predict import predict_customer

demo_risk = {
    "Recency": 395, "Frequency": 13, "MonetaryTotal": 1760.0, "MonetaryAvg": 130.0,
    "MonetaryStd": 13.0, "CustomerTenure": 443, "Age": 65, "SupportTickets": 11,
    "Satisfaction": 4.0, "WeekendRatio": 0.15, "ReturnRatio": 0.75,
    "UniqueProducts": 5, "CancelledTrans": 26, "RFMSegment": 0, "SpendingCat": 0, "ChurnRisk": 2
}

result = predict_customer(demo_risk)
print(f"Probabilité : {result['churn_proba']}")
print(f"Risque      : {result['risk_level']}")
print(f"Prédiction  : {result['churn_predicted']}")
print(f"Label       : {result['churn_label']}")
