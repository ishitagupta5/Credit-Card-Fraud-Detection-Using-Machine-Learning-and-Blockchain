#!/usr/bin/env python
"""
Trains an Isolation Forest on Kaggle's creditcard.csv, prints metrics,
and saves the fitted model as model.pkl.
"""

import joblib, pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

CSV_PATH   = "creditcard.csv"   # <-- uses the exact filename you asked for
MODEL_OUT  = "model.pkl"
TEST_SIZE  = 0.20
RANDOM_SEED = 42

def main():
    df = pd.read_csv(CSV_PATH)

    # -------- features & label --------
    X = df.drop(columns=["Class", "Time"])   # keep Amount + V1…V28
    y = df["Class"]                          # 1 = fraud, 0 = legit

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    clf = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=RANDOM_SEED
    ).fit(X_train)

    y_pred = np.where(clf.predict(X_test) == -1, 1, 0)

    print("Confusion matrix\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report\n", classification_report(y_test, y_pred, digits=4))

    joblib.dump(clf, MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

if __name__ == "__main__":
    main()
