# Aave V2 Wallet Credit Scoring System


An unsupervised machine learning pipeline that generates credit scores (0-1000) for Ethereum wallets based on their transaction history in Aave V2.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/your-username/aave-credit-scoring.git
cd aave-credit-scoring
pip install -r requirements.txt
Usage
Add your transaction data as src/data/transactions.json

Run the scoring pipeline:

bash
python src/main.py
📊 Outputs
File	Description
src/outputs/wallet_scores.csv	All wallet addresses with credit scores
src/outputs/score_distribution.png	Visual histogram of scores
analysis.md	Detailed report with insights
📈 Score Interpretation
Score	Risk Level	Typical Behavior
900-1000	Excellent	Consistent deposits, rare liquidations
700-900	Good	Occasional borrowing, healthy ratios
400-700	Moderate	Some liquidation history
0-400	High Risk	Bot-like patterns, frequent liquidations
🏗️ Project Structure
text
.
├── src/
│   ├── data/               # Raw transaction data
│   ├── models/             # Saved ML models
│   ├── outputs/            # Generated scores
│   ├── main.py             # Main pipeline
│   ├── features.py         # Feature engineering
│   └── model.py           # ML model code
├── analysis.md            # Auto-generated report
├── requirements.txt       # Dependencies
└── README.md             # This file
📝 Methodology
Data: 100K raw Aave V2 transactions

Features: 15+ metrics including:

Transaction frequency

Financial ratios

Liquidation history

Behavioral patterns

Model: Isolation Forest (unsupervised anomaly detection)
