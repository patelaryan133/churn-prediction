"""Customer Churn Prediction. Usage: python churn_prediction.py [Churn_Modelling.csv]"""
import sys, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "Churn_Modelling.csv")
tgt = "Exited" if "Exited" in df else "Churn"
df = df.drop(columns=[c for c in ["RowNumber","CustomerId","Surname","customerID"] if c in df])
df = df.dropna()
if df[tgt].dtype == object: df[tgt] = df[tgt].map({"Yes":1,"No":0})
X = pd.get_dummies(df.drop(columns=tgt), drop_first=True).astype(float)
y = df[tgt]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
for n, m in models.items():
    m.fit(Xtr_s, ytr)
    p = m.predict(Xte_s)
    print(f"\n=== {n} ===  Acc: {accuracy_score(yte,p):.4f}  ROC-AUC: {roc_auc_score(yte, m.predict_proba(Xte_s)[:,1]):.4f}")
    print(classification_report(yte, p, digits=4))
    if hasattr(m, "feature_importances_"):
        print("Top features:\n", pd.Series(m.feature_importances_, X.columns).nlargest(5).round(4).to_string())
