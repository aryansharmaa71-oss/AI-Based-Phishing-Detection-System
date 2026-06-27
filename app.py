import re
from urllib.parse import urlparse
import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AI Phishing Link Guardian")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load AI Model safely
try:
    model = joblib.load("phishing_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except FileNotFoundError:
    print("ERROR: Run train.py first to generate the model files!")

def extract_features(url: str) -> dict:
    """Extract features from raw URL mapping directly to your dataset's columns"""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
        
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    features = {}
    
    # 1. Have_IP
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}'
    features['Have_IP'] = 1 if re.match(ip_pattern, domain) else 0
    
    # 2. Have_At
    features['Have_At'] = 1 if '@' in url else 0
    
    # 3. URL_Length
    features['URL_Length'] = len(url)
    
    # 4. URL_Depth
    features['URL_Depth'] = len([x for x in parsed_url.path.split('/') if x])
    
    # 5. Redirection
    features['Redirection'] = 1 if url.rfind('//') > 7 else 0
    
    # 6. https_Domain
    features['https_Domain'] = 1 if 'https' in domain else 0
    
    # 7. TinyURL
    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs"
    features['TinyURL'] = 1 if re.search(shortening_services, url) else 0
    
    # 8. Prefix/Suffix (Hyphen in domain)
    features['Prefix/Suffix'] = 1 if '-' in domain else 0
    
    # Fill remaining columns with safe/average defaults for features hard to parse statically
    # (DNS_Record, Web_Traffic, Domain_Age, Domain_End, iFrame, Mouse_Over, Right_Click, Web_Forwards)
    features['DNS_Record'] = 0
    features['Web_Traffic'] = 0
    features['Domain_Age'] = 1
    features['Domain_End'] = 1
    features['iFrame'] = 0
    features['Mouse_Over'] = 0
    features['Right_Click'] = 0
    features['Web_Forwards'] = 0
    
    return features

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_url(request: Request, url: str = Form(...)):
    # 1. Extract features from user input
    extracted = extract_features(url)
    
    # 2. Construct DataFrame matching the exact training columns order
    input_df = pd.DataFrame([extracted])[feature_columns]
    
    # 3. Predict with AI Model
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    confidence = round(probabilities[prediction] * 100, 2)
    
    # Label mapping (adjust based on your dataset encoding, assuming 1=Phishing, 0=Safe)
    status = "MALICIOUS / PHISHING" if prediction == 1 else "SAFE / LEGITIMATE"
    result_class = "danger" if prediction == 1 else "success"
    
    return templates.TemplateResponse(request, "index.html", {
        "url": url,
        "status": status,
        "confidence": f"{confidence}%",
        "result_class": result_class
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)