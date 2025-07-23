import pandas as pd
from tqdm import tqdm

def calculate_wallet_features(wallet_transactions):
    """Calculate features for a single wallet"""
    features = {}
    
    # Total transactions
    features['total_txs'] = len(wallet_transactions)
    
    # Count of specific action types
    for action in ['deposit', 'borrow', 'repay', 'redeem', 'liquidation']:
        features[f'{action}_count'] = wallet_transactions['type'].str.lower().str.contains(action).sum()
    
    # Temporal features: first activity, last activity, transaction gaps
    timestamps_sorted = pd.to_datetime(wallet_transactions['block_timestamp']).sort_values()

    if len(timestamps_sorted) > 0:
        features['first_activity'] = timestamps_sorted.iloc[0]
        features['last_activity'] = timestamps_sorted.iloc[-1]
    else:
        features['first_activity'] = pd.NaT
        features['last_activity'] = pd.NaT

    if len(timestamps_sorted) > 1:
        time_deltas = timestamps_sorted.diff().dt.total_seconds().dropna()
        features['time_between_txs_mean'] = time_deltas.mean()
        features['time_between_txs_std'] = time_deltas.std()
    else:
        features['time_between_txs_mean'] = 0
        features['time_between_txs_std'] = 0

    # Financial features: total amounts and ratios
    deposits = wallet_transactions[wallet_transactions['type'].str.lower() == 'deposit']
    borrows = wallet_transactions[wallet_transactions['type'].str.lower() == 'borrow']

    features['total_deposit_value'] = deposits['amount_usd'].sum()
    features['total_borrow_value'] = borrows['amount_usd'].sum()
    
    # Safe division to prevent zero division error
    features['borrow_to_deposit_ratio'] = (
        features['total_borrow_value'] / (features['total_deposit_value'] + 1e-6)
    )

    return features

def generate_all_features(transactions_df):
    """Generate features for all wallets"""
    print("Generating features for all wallets...")
    grouped = transactions_df.groupby('user')
    
    features_list = []
    for wallet, txs in tqdm(grouped, desc="Processing wallets"):
        wallet_features = calculate_wallet_features(txs)
        wallet_features['wallet'] = wallet
        features_list.append(wallet_features)
    
    features_df = pd.DataFrame(features_list)
    return features_df
