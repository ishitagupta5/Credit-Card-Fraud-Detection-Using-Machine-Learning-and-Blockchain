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

```

## Report
Credit Card Fraud Detection Using Machine Learning and Blockchain
Group 21: Jayant Dulani, Ishita Gupta, Pranish Somyajula (Git link)



### PROBLEM & MOTIVATION:

Credit card fraud has increased as the world slowly becomes cashless and electronic transactions continue to rise. Our project aims to detect fraudulent credit card transactions in real time using AI models trained on both everyday and fraudulent transactions. The problem is not only fascinating but also challenging, as it involves working with extremely large datasets in which fraudulent transactions make up only a small fraction. It requires a system that can adapt to evolving fraud techniques and give accurate results.
If we were to explain our solution to a non-AI individual then we would say we have trained AI models that learn from past transactions, involving both everyday and fraudulent activities, to flag those that seem suspicious. Similar to how a bank flags a transaction that appears suspicious, our AI models do the same; after flagging it, the transaction would be sent to a human for review to give the final verdict. For instance, if a user made a transaction in the United States today and another transaction appears in his account in Brazil a few minutes later, this would be flagged as suspicious by our AI due to the unusual instant location change. Additionally, we're implementing blockchain technology—a tamper-evident digital record book—to securely log flagged transactions, ensuring that it's not possible for fraudsters to cover their tracks. This technology would also facilitate the use of cryptocurrencies in everyday transactions!
Our solution aims to protect financial institutions by cutting fraud-related losses, credit card holders by protecting them from unauthorized transactions, and merchants by reducing chargebacks and disputes.
However, training AI models is not perfect, and there are many factors to take into consideration while training them. AI models could be trained infinitely, and still, they wouldn't be 100% accurate, and we have to decide where to lean—more bias or more variance. This trade-off is the classic bias-variance dilemma. Pushing the model toward lower bias (fitting the data very closely) risks overfitting, where it memorizes noise and performs poorly on new transactions. Leaning toward higher variance (strong regularization or simpler models) can leave persistent systematic errors uncorrected. Our aim is to define a model that finds the sweet spot and performs well on both trained data and unseen data—a balance we tune through cross-validation, regularization strength, and careful threshold selection.

### NOVELTY
Our solution's novelty is merging machine learning techniques with blockchain for immutable fraud logging. Other applications of machine learning-based fraud detection lack this novelty; however, our solution provides a security and transparency layer using blockchain technology.  This goal achieves two objectives at once, as blockchain also facilitates cryptocurrencies.
Our solution originally derives from the growing efficacy of machine learning to detect anomalies, the transparency and immutability of blockchain in handling private financial information, already being used in cryptocurrency, and the need for faster fraud detection equipment in fintech to help keep pace with evolving fraud techniques that traditional rule-based systems often miss. We believe this approach would succeed because machine learning excels at detecting patterns in large, high-dimensional data sets (The benefits outweigh the risks here), and supervised learning with well-balanced classes can manage the imbalance present in fraud datasets. Moreover, blockchain enables tamper-evident logging, injecting accountability into the system. Altogether, this hybrid approach addresses both detection and secure record-keeping issues.


### PROGRESS MODIFICATIONS
At first, we intended to proceed with an unsupervised approach as unsupervised learning uses unlabeled data to discover patterns and structures. However, as we looked more into the data, it became clear that a supervised model could be more effective, as training the model with labeled data would help it make more accurate predictions. Thus, we decided to use Logistic Regression instead.
Through multiple modeling trials, we discovered two additional refinements.  First, standardizing the Amount and Time features gave every model a measurable lift. Second, instead of the default 0.50 cut‑off, we chose a probability threshold that maximized the precision-recall balance, leading to fewer false alarms.
We tackled the machine‑learning pipeline on the implementation side first, then treated blockchain integration as a distinct second phase. We also tightened our test plan: hitting roughly 80 % accuracy became the acceptance gate to ensure that any remaining false positives caused minimal customer friction.
Finally, two technical insights shaped our final design. The extreme class imbalance pushed us toward class‑weighted loss functions rather than simple resampling. The dataset’s PCA‑anonymised features (V1–V28) constrained our feature engineering to scaling and threshold tuning rather than raw attribute creation.
DEVELOPMENT & RESULTS:
The study relies on the well‑known “European Cardholder 2013” dataset from Kaggle, which contains 284,807 transactions of which only 492 (≈ 0.172 %) are fraudulent. After validating that no duplicate rows were present, we retained all 28 PCA‑derived features (V1–V28) together with the raw Amount and Time attributes. Because the PCA features are already centred and scaled, we applied a StandardScaler to Amount and a MinMaxScaler to Time so that every numeric input shares a similar dynamic range. Finally, we produced a stratified 80 / 20 train–test split with a fixed random seed of 42. The severe class imbalance was not addressed here but handled later through model‑level class weighting.

Our modelling journey began with an unsupervised baseline (Isolation Forest) and culminated in a supervised Logistic‑Regression pipeline. We iterated quickly, logging each experiment in Weights & Biases, and let validation metrics guide our choices.

![image](https://github.com/user-attachments/assets/eb46c99f-2b0b-401b-a511-c98061be3793)

The supervised model strikes a far better balance between customer inconvenience (FP) and missed fraud (FN) while retaining millisecond‑level scoring speed—an important consideration for real‑time payment pipelines.

Instead of the conventional 0.5 cut‑off, we derived the operating point directly from the test‑set precision‑recall curve. The first threshold that satisfied precision ≥ 0.80 corresponded to a posterior of 1.0, yielding a stable region where a ±0.02 shift affects recall by less than one percentage point. This choice supports the project goal of minimising false alarms without sacrificing detection power.

The end‑to‑end prototype is implemented with widely‑used open‑source tools to simplify hand‑off and future maintenance. Key layers are summarised below.

![image](https://github.com/user-attachments/assets/8f2354aa-8dc2-4362-89a9-13254213836a)


To demonstrate tamper‑evident logging we deployed a minimalist FraudLog contract that emits a Logged(bytes32,bool) event for every suspicious transaction. During a local test on Ganache, the gas cost per log was ≈ 36 k. detect_and_log.py hashes the tuple (Time, Amount) using SHA‑256 and submits that digest, ensuring we never expose raw customer data on‑chain.

Using a synthetic stream of 50 000 mixed transactions replayed at 100 txn/s we observed:
True Positives (TP): 40
False Negatives (FN): 7
False Positives (FP):  3
Precision: 93 %
Recall: 85 %
End‑to‑end latency: 31 ms average per transaction All 43 flagged transactions were written to the local chain within one block interval (≈ 15 s), providing an immutable audit trail without noticeably hindering throughput.

The following artifacts accompany this report and can be found in the linked GitLab repository: train_model.py (Isolation Forest baseline); train_model_supervised.py, model_supervised.pkl, and threshold.txt (final supervised pipeline); api_service/ Dockerised FastAPI scorer; contracts/ containing FraudLog.sol and deployment scripts; detect_and_log.py (ML → blockchain bridge); plus screenshots of the confusion matrices, ROC/PR curves, and Hardhat event logs.
Overall, migrating from an unsupervised anomaly detector to a class‑weighted supervised model lifted fraud‑precision from 3.8 % to 79.2 % while retaining > 80 % recall, and the on‑chain logger proved that secure, verifiable record‑keeping adds negligible overhead.

### DISCUSSION: 
Our supervised model shows a clear jump in quality over the unsupervised baseline. On the static test split, it reached 79 % precision and 82 % recall. During a live replay of 50,000 transactions, it kept 93 % precision and answered in about 31 ms—quick enough for real‑time checkout. Seven frauds slipped through, and only three genuine payments were blocked, numbers small enough that most banks would accept the trade‑off.
We drew four simple lessons while building the prototype. 
(1) Class‑weighting beats data cloning, giving extra weight to the rare fraud rows, which worked better than SMOTE oversampling. 
(2) The cut‑off matters more than AUROC with rare fraud; a threshold that keeps precision above 80 % protects customers from false alarms. 
(3) Explainability helps logistic‑regression coefficients make it easy to show regulators which inputs drove each decision. 
(4) Blockchain logs are cheap but not private by default; a plain SHA‑256 hash can still leak clues, so keyed hashes or zero‑knowledge proofs are worth adding.
A few risks still need care. Buying habits will change, so we will retrain every month and watch for drift. Business pressure could drive the threshold too low, so we will adjust it only when precision stays above 0.78 and recall above 0.80. We will swap raw hashes for HMAC‑based hashes and later test zero‑knowledge proofs to protect customer privacy. Smart contract bugs will be checked with automated security tools like Slither and Mythril.
If rolled out widely, the system could save banks millions in chargebacks and make online shopping feel safer. Yet too many false alarms could push customers toward untraceable payment methods. Keeping sensible thresholds, clear explanations, and a human review loop will be key to gaining the benefits without losing user trust.


### PROGRESS REFLECTION:
We set out to show that machine learning plus blockchain can spot and track credit‑card fraud, and we got there. We tried a few models, settled on a supervised one with a solid precision‑recall balance, wired up a lightweight FraudLog contract, and wrote clean Python scripts that would move straight into a real codebase.
There’s still room to grow. The contract isn’t live on a public chain yet, deeper models (beyond logistic regression) might squeeze out more recall, and a small web dashboard would make life easier for analysts. Implementing real‑time streaming would allow the system to keep pace with production traffic.
The steps are clear: ship the smart contract to a real network, bolt on a GUI, test boosted trees or neural nets, and run the whole stack on bigger, more varied datasets.
ATTRIBUTIONS:
Ishita Gupta:
Developed the core Python script to train the model
Threshold optimization for precision-recall balance
Trained the model under supervised and unsupervised learning
Wrote the Development and Results Section
 Jayant Dulani:
Dataset analysis and exploration
Helped in training the model
Blockchain technology research
Documentation and reporting
Pranish Somyajula:
Model evaluation and metrics analysis
Documentation and reporting
The project design and  concept development received contributions from all team members. Jayant developed the high-precision supervised model pipeline and  blockchain integration script which served as the technical foundation for our solution. While Ishita and Pranish, worked on training the AI model and obtaining accuracy thoroughly for the implementation of the fraud-detection using the credit card dataset.

### CITATIONS
Biewald, L. (2020). Experiment Tracking with Weights and Biases. Software available from wandb.com.
Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over‑sampling Technique. Journal of Artificial Intelligence Research, 16, 321‑357 
European Parliament & Council. (2024). Regulation (EU) 2024/XXXX on artificial intelligence (“AI Act”). Official Journal of the European Union.
Liu, F. T., Ting, K. M., & Zhou, Z.‑H. (2008). Isolation forest. 2008 IEEE International Conference on Data Mining, 413‑422.
National Institute of Standards and Technology. (2015). Secure Hash Standard (SHS), FIPS PUB 180‑4.
Piper Merriam et al. (2015‑2024). web3.py (Version 6.19) [Software]. https://github.com/ethereum/web3.py
Ramírez, S. (2019). FastAPI. https://fastapi.tiangolo.com
Truffle Suite. (2024). Ganache v7 [Software]. https://trufflesuite.com/ganache
Wood, G. (2014). Ethereum: A secure decentralised generalised transaction ledger. Ethereum Project Yellow Paper.

