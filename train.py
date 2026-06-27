import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    print("Loading dataset...")
    df = pd.read_csv("5.urldata.csv")
    
    # Separate features and target label
    X = df.drop(['Domain', 'Label'], axis=1)
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save both the model and the expected feature names
    joblib.dump(model, "phishing_model.pkl")
    joblib.dump(X.columns.tolist(), "feature_columns.pkl")
    print("Model and features successfully saved!")

if __name__ == "__main__":
    train_model()