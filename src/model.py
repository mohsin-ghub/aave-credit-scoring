# src/model.py
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def train_model(features_df):
    """Train isolation forest model for anomaly detection"""
    print("Training model...")
    
    # Select features (excluding wallet address)
    X = features_df.drop(columns=['wallet'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # Expected proportion of outliers
        random_state=42
    )
    model.fit(X_scaled)
    
    return model, scaler

def predict_scores(model, scaler, features_df):
    """Generate scores from features"""
    X = features_df.drop(columns=['wallet'])
    X_scaled = scaler.transform(X)
    
    # Get anomaly scores (-1 to 1 where -1 is outlier)
    raw_scores = model.decision_function(X_scaled)
    
    # Convert to 0-1000 scale
    min_score, max_score = raw_scores.min(), raw_scores.max()
    normalized_scores = 1000 * (raw_scores - min_score) / (max_score - min_score)
    
    return normalized_scores

def save_model(model, scaler, path='src/models'):
    """Save model and scaler"""
    joblib.dump(model, f'{path}/model.pkl')
    joblib.dump(scaler, f'{path}/scaler.pkl')

def load_model(path='src/models'):
    """Load model and scaler"""
    model = joblib.load(f'{path}/model.pkl')
    scaler = joblib.load(f'{path}/scaler.pkl')
    return model, scaler