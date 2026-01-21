"""
Breast Cancer Prediction Model Development
This script develops a machine learning model to predict breast cancer diagnosis
using the Breast Cancer Wisconsin (Diagnostic) dataset.

DISCLAIMER: This system is strictly for educational purposes and must not be 
presented as a medical diagnostic tool.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def load_dataset():
    """Load the Breast Cancer Wisconsin dataset"""
    print("Loading Breast Cancer Wisconsin dataset...")
    try:
        # Load dataset from sklearn
        cancer_data = load_breast_cancer()
        X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
        y = pd.Series(cancer_data.target, name='diagnosis')
        
        # Map diagnosis: 1 = Malignant, 0 = Benign (sklearn convention)
        diagnosis_map = {0: 'Malignant', 1: 'Benign'}
        y_labels = y.map(diagnosis_map)
        
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(X)}")
        print(f"Total features: {X.shape[1]}")
        print(f"Diagnosis distribution:\n{y_labels.value_counts()}")
        
        return X, y, y_labels
        
    except Exception as e:
        print(f"ERROR loading dataset: {str(e)}")
        raise

def preprocess_data(X, y, y_labels):
    """
    Perform data preprocessing:
    - Handling missing values
    - Feature selection (5 features from recommended 8)
    - Encoding target variable
    - Feature scaling
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    # Selected 5 features from the recommended 8:
    # Note: Dataset uses spaces in column names
    selected_features = ['mean radius', 'mean texture', 'mean area', 
                         'mean smoothness', 'mean compactness']
    
    print(f"\nSelected Features (5 of 8):")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    
    print(f"\nTarget: diagnosis (Benign / Malignant)")
    
    # Select only the features we need
    X_selected = X[selected_features].copy()
    
    print(f"\nMissing values check:")
    missing = X_selected.isnull().sum()
    if missing.sum() > 0:
        print(f"  Found missing values: {missing[missing > 0].to_dict()}")
        X_selected = X_selected.dropna()
    else:
        print("  No missing values found")
    
    print(f"\nDataset statistics:")
    print(f"  Features shape: {X_selected.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Encode target variable
    print(f"\nEncoding target variable:")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Encoding: 0 = Malignant, 1 = Benign")
    
    # Save label encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder.joblib')
    
    # Feature scaling (mandatory for distance-based and regularized models)
    print(f"\nApplying feature scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print(f"\nFeature scaling statistics (after scaling):")
    print(f"  Mean (should be ~0): {X_scaled.mean().mean():.6f}")
    print(f"  Std (should be ~1): {X_scaled.std().mean():.6f}")
    
    return X_scaled, y_encoded, selected_features

def train_model(X, y):
    """
    Train Logistic Regression model for binary classification
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split (80-20 train-test with stratification):")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # Create and train the model
    print(f"\nAlgorithm: Logistic Regression")
    print(f"  Solver: lbfgs")
    print(f"  Max iterations: 5000")
    
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=5000,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print(f"\nModel training completed!")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using classification metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {cm[0][0]}")
    print(f"  False Positives (FP): {cm[0][1]}")
    print(f"  False Negatives (FN): {cm[1][0]}")
    print(f"  True Positives (TP):  {cm[1][1]}")
    
    # Detailed Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

def save_model(model, metrics):
    """
    Save the trained model and metrics to disk
    """
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'models/cancer_model.joblib')
    print(f"\n✓ Model saved: models/cancer_model.joblib")
    
    # Save metrics
    joblib.dump(metrics, 'models/model_metrics.joblib')
    print(f"✓ Metrics saved: models/model_metrics.joblib")
    
    # Save feature names for later use
    feature_names = ['radius_mean', 'texture_mean', 'area_mean', 
                     'smoothness_mean', 'compactness_mean']
    joblib.dump(feature_names, 'models/feature_names.joblib')
    print(f"✓ Feature names saved: models/feature_names.joblib")

def test_model_reload():
    """
    Test that the saved model can be reloaded and used for prediction
    """
    print("\n" + "="*60)
    print("TESTING MODEL RELOAD")
    print("="*60)
    
    try:
        # Reload model
        print("\nReloading saved model...")
        model = joblib.load('models/cancer_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        le = joblib.load('models/label_encoder.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        
        print("✓ Model reloaded successfully")
        print(f"✓ Scaler reloaded successfully")
        print(f"✓ Label encoder reloaded successfully")
        print(f"✓ Feature names reloaded: {feature_names}")
        
        # Test prediction with sample data
        print(f"\nTesting prediction with sample data...")
        sample_data = np.array([[13.0, 20.0, 82.0, 0.08, 0.07]])
        sample_scaled = scaler.transform(sample_data)
        prediction = model.predict(sample_scaled)
        prediction_proba = model.predict_proba(sample_scaled)
        
        diagnosis = le.inverse_transform(prediction)[0]
        confidence = max(prediction_proba[0]) * 100
        
        print(f"  Sample input: {sample_data[0]}")
        print(f"  Predicted diagnosis: {diagnosis}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"\n✓ Model is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reloading model: {str(e)}")
        return False

def main():
    """Main execution function"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  BREAST CANCER PREDICTION MODEL DEVELOPMENT".center(58) + "║")
    print("║" + " "*58 + "║")
    print("║" + "  EDUCATIONAL PURPOSES ONLY - NOT FOR MEDICAL USE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # Load dataset
        X, y, y_labels = load_dataset()
        
        # Preprocess data
        X_processed, y_encoded, features = preprocess_data(X, y, y_labels)
        
        # Train model
        model, X_test, y_test = train_model(X_processed, y_encoded)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        save_model(model, metrics)
        
        # Test model reload
        test_model_reload()
        
        print("\n" + "="*60)
        print("✓ MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now use the trained model in the Flask app.")
        print("Run: python app.py")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
