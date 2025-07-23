# 🧠 Aave V2 Wallet Credit Scoring

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An unsupervised machine learning pipeline that assigns credit scores (ranging from 0 to 1000) to Ethereum wallets based on their Aave V2 transaction history. This project helps identify wallet behavior patterns such as liquidations, borrowing activity, and financial responsibility without needing labeled training data.

---

## 🚀 Quick Start

### ✅ Prerequisites

- Python 3.8+
- `pip` package manager
- A JSON file containing Aave V2 wallet transaction data

### 🔧 Installation

```bash
git clone https://github.com/your-username/aave-credit-scoring.git
cd aave-credit-scoring
pip install -r requirements.txt
📂 Add Your Data
Place your Aave V2 transaction data inside:

bash
src/data/transactions.json

▶️ Usage

Run the pipeline using:

bash
python src/main.py

This will generate features, apply the ML model, and export credit scores with visual analysis.

📊 Outputs
File	Description
src/outputs/wallet_scores.csv	Wallet addresses with their credit scores
src/outputs/score_distribution.png	Histogram of the credit score distribution
analysis.md	Auto-generated report with observations


📈 Score Interpretation
Score Range	Risk Level	Wallet Behavior
900 - 1000	Excellent	Active, stable, rarely liquidated
700 - 900	Good	Moderate borrowing, healthy ratios
400 - 700	Moderate	Some liquidation or irregular activity
0 - 400	High Risk	Frequent liquidation, bot-like behavior

🏗️ Project Structure
├── src/
│   ├── data/               # Input: Raw transaction data
│   ├── models/             # Saved ML models
│   ├── outputs/            # Generated scores and visualizations
│   ├── main.py             # Main pipeline entry point
│   ├── features.py         # Feature engineering logic
│   ├── model.py            # ML model definition
│   └── analysis.py         # Post-processing and insights
├── analysis.md             # Auto-generated analysis report
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


🧪 Methodology
Data: ~100K Aave V2 transactions

Feature Engineering:

Wallet activity over time

Borrow-to-collateral ratios

Number and size of liquidations

Behavioral consistency

Model:

Isolation Forest (unsupervised anomaly detection)

Score scaled from 0 to 1000 based on anomaly level

📄 License
This project is licensed under the MIT License.
