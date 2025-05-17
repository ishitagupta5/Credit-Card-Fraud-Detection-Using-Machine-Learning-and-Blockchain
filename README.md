# AI_Capstone

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

## Setup (≈5 min)

```bash
# 1 – Python deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2 – Dataset

# 3 – Train (~1 min)
python train_model.py              # writes model.pkl

# 4 – Local blockchain (Ganache or Hardhat)
npm install --global hardhat
npx hardhat compile
npx hardhat node &                 # local RPC at :8545
npx hardhat run scripts/deploy.js --network localhost
#   copy the printed contract address

# 5 – Detect & log (sample 1000 rows)
python detect_and_log.py --contract 0xABC... --sample 1000
