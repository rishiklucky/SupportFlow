import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Configuration
DATA_FILE = "support_tickets_data.csv"
MODEL_DIR = "models"
CATEGORY_MODEL_FILE = f"{MODEL_DIR}/category_model.pkl"
URGENCY_MODEL_FILE = f"{MODEL_DIR}/urgency_model.pkl"

def train():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("Data file not found. Please run distribute_data.py first.")
        return

    # Basic preprocessing
    df['text'] = df['text'].fillna('')

    X = df['text']
    y_category = df['category']
    y_urgency = df['urgency']

    # Split data
    X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
        X, y_category, y_urgency, test_size=0.2, random_state=42
    )

    # --- Train Category Model ---
    print("\nTraining Category Model...")
    category_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english', 
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            lowercase=True
        )),
        ('clf', LogisticRegression(
            random_state=42, 
            max_iter=2000, 
            class_weight='balanced',
            C=0.5,
            solver='lbfgs'
        ))
    ])
    
    category_pipeline.fit(X_train, y_cat_train)
    
    y_cat_pred = category_pipeline.predict(X_test)
    cat_accuracy = accuracy_score(y_cat_test, y_cat_pred)
    print(f"\nCategory Model Accuracy: {cat_accuracy:.2%}")
    print("Category Classification Report:")
    print(classification_report(y_cat_test, y_cat_pred))

    # --- Train Urgency Model ---
    print("\nTraining Urgency Model...")
    urgency_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english', 
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            lowercase=True
        )),
        ('clf', GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
    
    urgency_pipeline.fit(X_train, y_urg_train)
    
    y_urg_pred = urgency_pipeline.predict(X_test)
    urg_accuracy = accuracy_score(y_urg_test, y_urg_pred)
    print(f"\nUrgency Model Accuracy: {urg_accuracy:.2%}")
    print("Urgency Classification Report:")
    print(classification_report(y_urg_test, y_urg_pred))

    # --- Save Models ---
    import os
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(category_pipeline, CATEGORY_MODEL_FILE)
    joblib.dump(urgency_pipeline, URGENCY_MODEL_FILE)
    print(f"\nModels saved to {MODEL_DIR}/")
    print(f"\nSummary:")
    print(f"Category Accuracy: {cat_accuracy:.2%}")
    print(f"Urgency Accuracy: {urg_accuracy:.2%}")

if __name__ == "__main__":
    train()
