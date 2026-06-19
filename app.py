import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import tldextract
from xgboost import XGBClassifier

app = FastAPI(title="AI Phishing Detection API")

# Enable CORS so your frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = XGBClassifier()

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(url: str):
    url = str(url).strip().lower()
    ext = tldextract.extract(url)
    
    subdomains = ext.subdomain.split('.') if ext.subdomain else []
    count_subdomains = len([s for s in subdomains if s])

    return {
        'url_length': len(url),
        'hostname_length': len(ext.fqdn),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'count_at': url.count('@'),
        'count_question': url.count('?'),
        'count_equal': url.count('='),
        'count_slash': url.count('/'),
        'has_http': 1 if 'http://' in url else 0,
        'has_https': 1 if 'https://' in url else 0,
        'is_ip': 1 if ext.domain.replace('.', '').isdigit() else 0,
        'count_subdomains': count_subdomains
    }

# ==========================================
# MODEL INITIALIZATION ON STARTUP
# ==========================================
@app.on_event("startup")
def train_model():
    global model
    print("[-] Training AI Model...")
    
    # Mock data setup (Replace with a larger dataset CSV file for production)
    mock_data = [
        {"url": "https://www.google.com", "label": 0},
        {"url": "https://www.github.com", "label": 0},
        {"url": "https://www.amazon.com", "label": 0},
        {"url": "http://secure-login-paypal-verify.com", "label": 1},
        {"url": "http://192.168.1.105/login.php", "label": 1},
        {"url": "https://amaz0n-security-check.net", "label": 1}
    ] * 10
    
    df = pd.DataFrame(mock_data)
    feature_list = df['url'].apply(extract_features).tolist()
    X = pd.DataFrame(feature_list)
    y = df['label']
    
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    print("[+] Model trained successfully and ready!")

# ==========================================
# API ENDPOINTS
# ==========================================
class URLRequest(BaseModel):
    url: str

@app.post("/predict")
def predict_phishing(payload: URLRequest):
    if not payload.url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
        
    try:
        # Extract features and convert to DataFrame matching model expectations
        features = extract_features(payload.url)
        features_df = pd.DataFrame([features])
        
        # Predict
        prediction = int(model.predict(features_df)[0])
        probability = float(model.predict_proba(features_df)[0][1])
        
        return {
            "url": payload.url,
            "is_phishing": prediction == 1,
            "confidence_score": round(probability * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)