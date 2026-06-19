import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ==========================================
# 1. FEATURE EXTRACTION FUNCTION
# ==========================================
def extract_features(url):
    """
    Extracts lexical features from a given URL string.
    Returns a dictionary of features.
    """
    features = {}
    
    # Clean and parse URL
    url = str(url).strip().lower()
    ext = tldextract.extract(url)
    
    # 1. Length-based features
    features['url_length'] = len(url)
    features['hostname_length'] = len(ext.fqdn)
    
    # 2. Count of risky characters
    features['count_dots'] = url.count('.')
    features['count_hyphens'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_equal'] = url.count('=')
    features['count_slash'] = url.count('/')
    
    # 3. Binary indicators
    features['has_http'] = 1 if 'http://' in url else 0
    features['has_https'] = 1 if 'https://' in url else 0
    features['is_ip'] = 1 if ext.domain.replace('.', '').isdigit() else 0
    
    # 4. Subdomain count
    # e.g., 'sub.login.example.com' -> subdomains are ['sub', 'login']
    subdomains = ext.subdomain.split('.') if ext.subdomain else []
    features['count_subdomains'] = len([s for s in subdomains if s])

    return features

# ==========================================
# 2. SAMPLE DATA GENERATION
# ==========================================
def load_mock_data():
    """
    Generates a mock dataset for demonstration purposes.
    Replace this with a real dataset like the 'UCI Phishing Dataset' or Kaggle datasets.
    """
    # 0 = Legitimate, 1 = Phishing
    data = [
        {"url": "https://www.google.com", "label": 0},
        {"url": "https://www.github.com/login", "label": 0},
        {"url": "https://www.amazon.com/gp/css/homepage.html", "label": 0},
        {"url": "http://secure-login-paypal-verify.com/account", "label": 1},
        {"url": "http://192.168.1.105/index.php?update=true", "label": 1},
        {"url": "https://amaz0n-security-check.net/signin", "label": 1},
        {"url": "http://verify-your-identity-apple.com", "label": 1},
        {"url": "https://www.wikipedia.org", "label": 0}
    ]
    # Replicate data to give the model enough rows to run without errors
    return pd.DataFrame(data * 10) 

# ==========================================
# 3. MAIN TRAINING & EVALUATION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("[-] Loading dataset...")
    df = load_mock_data()
    
    print("[-] Extracting features from URLs...")
    # Apply feature extraction to every row
    feature_list = df['url'].apply(extract_features).tolist()
    X = pd.DataFrame(feature_list)
    y = df['label']
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[-] Training XGBoost Model...")
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # ==========================================
    # 4. LIVE PREDICTION TEST
    # ==========================================
    print("\n=== LIVE TEST ===")
    test_urls = [
        "https://www.microsoft.com",
        "http://update-netflix-billing-info.com/login"
    ]
    
    for test_url in test_urls:
        # Extract features for the new URL
        features_dict = extract_features(test_url)
        features_df = pd.DataFrame([features_dict])
        
        # Predict
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]
        
        status = "🚨 PHISHING" if prediction == 1 else "✅ LEGITIMATE"
        print(f"URL: {test_url}")
        print(f"Result: {status} (Phishing Risk: {probability*100:.1f}%)\n")