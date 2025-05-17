#!/usr/bin/env python
"""
detect_and_log.py
-----------------
Loads a *supervised* fraud‑detection model (model_supervised.pkl),
applies it to a sample from creditcard.csv, and logs each high‑confidence
fraud hash to the deployed FraudLog smart‑contract.

Run:
  python detect_and_log.py --contract 0xABC... --sample 1000
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from web3 import Web3


# ---------- helpers --------------------------------------------------------- #
def sha256_bytes(text: str) -> bytes:
    """Return 32‑byte SHA‑256 digest for a given string."""
    return hashlib.sha256(text.encode()).digest()


# ---------- main ------------------------------------------------------------ #
def main() -> None:
    # -------- arg‑parse ----------------------------------------------------- #
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default="creditcard.csv",
                    help="Input CSV (Kaggle credit‑card dataset)")
    ap.add_argument("--model",  default="model_supervised.pkl",
                    help="Fitted sklearn pipeline (from train_model_supervised.py)")
    ap.add_argument("--contract", required=True,
                    help="Deployed FraudLog contract address")
    ap.add_argument("--sample", type=int, default=1000,
                    help="Rows to sample for a quick demo (use ‑1 for full)")
    ap.add_argument("--rpc",    default="http://127.0.0.1:8545",
                    help="JSON‑RPC endpoint (Hardhat/Ganache)")
    args = ap.parse_args()

    # -------- load data ----------------------------------------------------- #
    df_full = pd.read_csv(args.csv)
    df = df_full if args.sample == -1 else df_full.sample(n=args.sample, random_state=1)
    X = df.drop(columns=["Class", "Time"])

    # -------- load model + threshold --------------------------------------- #
    clf = joblib.load(args.model)         # pipeline with StandardScaler + LogisticReg
    thr_path = Path("threshold.txt")
    if not thr_path.exists():
        raise FileNotFoundError("threshold.txt not found – run train_model_supervised.py first")
    THR = float(thr_path.read_text().strip())

    # predict probability for class 1 (fraud)
    probs = clf.predict_proba(X)[:, 1]
    fraud_mask = probs >= THR                   # only high‑confidence frauds
    frauds = df[fraud_mask]

    print(f"Flagged {len(frauds)} / {len(df)} rows as fraud "
          f"(threshold = {THR:.4f})")

    # -------- connect to blockchain ---------------------------------------- #
    w3 = Web3(Web3.HTTPProvider(args.rpc))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot reach RPC at {args.rpc}")

    acct = w3.eth.accounts[0]                  # first local account

    # load ABI produced by Hardhat compile
    abi_file = Path("artifacts/contracts/FraudLog.sol/FraudLog.json")
    abi = json.loads(abi_file.read_text())["abi"]
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(args.contract),
        abi=abi
    )

    # dummy key (replace with your account’s key or unlock the account)
    PRIVATE_KEY = "0x" + "11" * 32
    nonce = w3.eth.get_transaction_count(acct)

    # -------- log each fraud ----------------------------------------------- #
    for _, row in frauds.iterrows():
        raw = f"{row.Time}_{row.Amount}"
        tx_hash_bytes = sha256_bytes(raw)

        txn = contract.functions.log(tx_hash_bytes, True).build_transaction({
            "from": acct,
            "nonce": nonce,
            "gas": 100000
        })
        signed = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_sent = w3.eth.send_raw_transaction(signed.rawTransaction)
        print("  →", tx_sent.hex()[:12], "... logged")

        nonce += 1


# ---------- entry‑point ----------------------------------------------------- #
if __name__ == "__main__":
    main()
