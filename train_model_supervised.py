#!/usr/bin/env python
"""
Supervised upgrade for the capstone prototype.

• Standardizes features
• Trains Logistic Regression with class_weight='balanced'
• Chooses a probability threshold that hits ≈80 % precision
• Saves both the fitted model and the threshold
"""

import joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report

CSV_PATH   = "creditcard.csv"
MODEL_OUT  = "model_supervised.pkl"
THR_OUT    = "threshold.txt"
TEST_SIZE  = 0.2
SEED       = 42

def main():
    df = pd.read_csv(CSV_PATH)

    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    pipe = make_pipeline(
        StandardScaler(with_mean=False),       # sparse‑safe; dataset is dense anyway
        LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=400,
            solver="lbfgs",
            n_jobs=-1,
            random_state=SEED,
        ),
    )

    pipe.fit(X_train, y_train)

    # ---- pick a threshold for ≈80 % precision ----
    probs = pipe.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, probs)

    # use the first threshold where precision ≥0.80
    try:
        best_idx = next(i for i, p in enumerate(prec) if p >= 0.80)
        best_thr = thr[best_idx - 1]          # thr array is one shorter
    except StopIteration:
        best_thr = 0.5                        # fallback

    # evaluate at chosen threshold
    y_pred = (probs >= best_thr).astype(int)
    print("Chosen threshold:", round(best_thr, 4))
    print("\nClassification report @ chosen threshold\n",
          classification_report(y_test, y_pred, digits=4))

    # persist
    joblib.dump(pipe, MODEL_OUT)
    Path(THR_OUT).write_text(str(best_thr))
    print(f"\nModel  → {MODEL_OUT}\nThreshold → {THR_OUT}")

if __name__ == "__main__":
    main()
