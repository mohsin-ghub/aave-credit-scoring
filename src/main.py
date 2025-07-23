#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

def calculate_entropy(timestamps):
    """Calculate entropy of transaction timestamps"""
    if len(timestamps) < 2:
        return 0
    deltas = np.diff(sorted(timestamps))
    hist = np.histogram(deltas, bins=10)[0]
    return entropy(hist / hist.sum())

def load_data(filepath):
    """Load and validate transaction data with robust error handling"""
    print(f"\n🔍 Loading data from {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {os.path.abspath(filepath)}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} transactions")
        return pd.DataFrame(data)
    except json.JSONDecodeError:
        print("⚠️ Standard JSON load failed, trying line-by-line...")
        data = []
        with open(filepath, 'r') as f:
            for line in tqdm(f, desc="Processing lines"):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"ℹ️ Loaded {len(data)} valid transactions")
        return pd.DataFrame(data)

def calculate_wallet_features(wallet_transactions):
    """Enhanced feature calculation with financial metrics and bot detection"""
    features = {
        'total_txs': len(wallet_transactions),
    }
    
    # Extract from actionData if available
    if 'actionData' in wallet_transactions.columns:
        try:
            amounts = wallet_transactions['actionData'].apply(
                lambda x: float(x['amount']) * float(x.get('assetPriceUSD', 1))
            )
            features.update({
                'total_usd_volume': amounts.sum(),
                'avg_usd_amount': amounts.mean(),
                'max_usd_amount': amounts.max()
            })
        except Exception as e:
            print(f"⚠️ Error processing amounts: {str(e)}")

    # Temporal features
    timestamp_col = next((col for col in ['timestamp', 'block_timestamp', 'createdAt'] 
                         if col in wallet_transactions.columns), None)
    
    if timestamp_col:
        timestamps = pd.to_datetime(wallet_transactions[timestamp_col])
        features.update({
            'first_activity': timestamps.min(),
            'last_activity': timestamps.max(),
            'tx_entropy': calculate_entropy(timestamps.view('int64'))
        })

    # Action counts and ratios
    action_col = next((col for col in ['action', 'type'] 
                      if col in wallet_transactions.columns), None)
    
    if action_col:
        for action in ['deposit', 'borrow', 'repay', 'redeem', 'liquidation']:
            features[f'{action}_count'] = wallet_transactions[action_col].str.lower().str.contains(action).sum()
        
        # Financial health ratios
        if 'total_usd_volume' in features:
            features['borrow_ratio'] = (
                features.get('borrow_count', 0) / 
                (features.get('deposit_count', 0) + 1e-6)
            )

    # Bot-like behavior detection
    if 'txHash' in wallet_transactions.columns:
        features['unique_tx_senders'] = wallet_transactions['txHash'].nunique()

    return features

def generate_features(transactions_df):
    """Generate features for all wallets with enhanced logging"""
    print("\n🔧 Generating features...")
    
    wallet_col = next((col for col in ['userWallet', 'user', 'address', 'userId'] 
                      if col in transactions_df.columns), None)
    
    if not wallet_col:
        raise KeyError(f"No wallet identifier found in: {transactions_df.columns.tolist()}")

    features = []
    grouped = transactions_df.groupby(wallet_col)
    
    for wallet, txs in tqdm(grouped, desc="Processing wallets"):
        try:
            features.append({
                'wallet': wallet,
                **calculate_wallet_features(txs)
            })
        except Exception as e:
            print(f"⚠️ Error processing {wallet}: {str(e)}")
            continue
    
    features_df = pd.DataFrame(features)
    
    # Post-processing
    if {'first_activity', 'last_activity'}.issubset(features_df.columns):
        features_df['wallet_age_days'] = (
            pd.to_datetime(features_df['last_activity']) - 
            pd.to_datetime(features_df['first_activity'])
        ).dt.total_seconds() / 86400
    
    print("\n✅ Feature summary:")
    print(features_df.describe().to_markdown())
    return features_df

def train_model(features_df):
    """Train credit model with feature importance tracking"""
    print("\n🤖 Training model...")
    
    exclude_cols = ['wallet', 'first_activity', 'last_activity']
    numeric_cols = [col for col in features_df.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(features_df[col])]
    
    X = features_df[numeric_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        verbose=1
    )
    model.fit(X_scaled)
    
    # Save artifacts
    os.makedirs('src/models', exist_ok=True)
    joblib.dump(model, 'src/models/credit_model.pkl')
    joblib.dump(scaler, 'src/models/scaler.pkl')
    
    return model, scaler, numeric_cols

def generate_scores(features_df, model, scaler, feature_names):
    """Generate scores with interpretability"""
    X = features_df[feature_names].fillna(0)
    X_scaled = scaler.transform(X)
    
    raw_scores = model.decision_function(X_scaled)
    scores = 1000 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    
    return np.clip(scores, 0, 1000)

def generate_analysis(scores, features_df):
    """Enhanced analysis with feature correlations"""
    print("\n📊 Generating analysis...")
    
    # Score distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(scores, bins=20, kde=True)
    plt.title('Wallet Credit Score Distribution')
    plt.savefig('src/outputs/score_distribution.png')
    plt.close()
    
    # Feature correlations
    corr_df = features_df.select_dtypes(include=np.number).corrwith(pd.Series(scores, name='credit_score'))
    
    # Save report
    with open('analysis.md', 'w') as f:
        f.write("# Aave V2 Credit Score Report\n\n")
        f.write("## Score Distribution\n")
        f.write("![Distribution](src/outputs/score_distribution.png)\n\n")
        f.write("## Feature Correlations\n")
        f.write(corr_df.sort_values().to_markdown())
        f.write("\n\n## Score Interpretation\n")
        f.write("| Score Range | Behavior Profile |\n")
        f.write("|-------------|------------------|\n")
        f.write("| 900-1000 | Ideal borrowers: Consistent deposits, low liquidation risk |\n")
        f.write("| 700-900 | Reliable users: Healthy financial ratios |\n")
        f.write("| 400-700 | Moderate risk: Occasional liquidations |\n")
        f.write("| 0-400 | High risk: Potential bots or exploit patterns |\n")

def main():
    try:
        # Pipeline
        df = load_data('src/data/transactions.json')
        features_df = generate_features(df)
        model, scaler, feature_names = train_model(features_df)
        scores = generate_scores(features_df, model, scaler, feature_names)
        
        # Save results
        os.makedirs('src/outputs', exist_ok=True)
        pd.DataFrame({
            'wallet': features_df['wallet'],
            'credit_score': scores
        }).to_csv('src/outputs/wallet_scores.csv', index=False)
        
        generate_analysis(scores, features_df)
        
        print("\n🎉 Pipeline completed!")
        print(f"📄 Scores saved to: src/outputs/wallet_scores.csv")
        print(f"📊 Analysis saved to: analysis.md")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()