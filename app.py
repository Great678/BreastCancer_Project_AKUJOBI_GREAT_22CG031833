"""
Breast Cancer Prediction Web Application
Flask application for breast cancer diagnosis prediction using trained ML model

DISCLAIMER: This system is strictly for educational purposes and must not be 
presented as a medical diagnostic tool.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import traceback

app = Flask(__name__)

# Global variables for model and preprocessing tools
model = None
scaler = None
label_encoder = None
feature_names = None
metrics = None

def load_model_artifacts():
    """Load the trained model and preprocessing artifacts"""
    global model, scaler, label_encoder, feature_names, metrics
    
    try:
        # Check if model files exist
        model_path = 'models/cancer_model.joblib'
        scaler_path = 'models/scaler.joblib'
        le_path = 'models/label_encoder.joblib'
        features_path = 'models/feature_names.joblib'
        metrics_path = 'models/model_metrics.joblib'
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            print("Please run: python model_development.py")
            return False
        
        print("Loading model artifacts...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(le_path)
        feature_names = joblib.load(features_path)
        metrics = joblib.load(metrics_path)
        
        print("✓ Model loaded successfully")
        return True
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Render the main page"""
    if model is None:
        return "Error: Model not loaded. Please run model_development.py first.", 500
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on user input
    """
    try:
        data = request.get_json()
        
        # Extract features from request
        mean_radius = float(data.get('radius_mean'))
        mean_texture = float(data.get('texture_mean'))
        mean_area = float(data.get('area_mean'))
        mean_smoothness = float(data.get('smoothness_mean'))
        mean_compactness = float(data.get('compactness_mean'))
        
        # Validate inputs
        if not all([mean_radius, mean_texture, mean_area, mean_smoothness, mean_compactness]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Validate ranges (these are approximate ranges based on dataset)
        if mean_radius < 0 or mean_radius > 50:
            return jsonify({'error': 'Radius Mean must be between 0 and 50'}), 400
        if mean_texture < 0 or mean_texture > 50:
            return jsonify({'error': 'Texture Mean must be between 0 and 50'}), 400
        if mean_area < 0 or mean_area > 3000:
            return jsonify({'error': 'Area Mean must be between 0 and 3000'}), 400
        if mean_smoothness < 0 or mean_smoothness > 1:
            return jsonify({'error': 'Smoothness Mean must be between 0 and 1'}), 400
        if mean_compactness < 0 or mean_compactness > 1:
            return jsonify({'error': 'Compactness Mean must be between 0 and 1'}), 400
        
        # Prepare features for prediction (in the correct order)
        features = np.array([[
            mean_radius,
            mean_texture,
            mean_area,
            mean_smoothness,
            mean_compactness
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get diagnosis and confidence
        diagnosis = str(label_encoder.inverse_transform([prediction])[0])
        confidence = float(max(prediction_proba) * 100)
        
        return jsonify({
            'success': True,
            'prediction': diagnosis,
            'confidence': confidence,
            'malignant_prob': float(prediction_proba[0]) * 100,
            'benign_prob': float(prediction_proba[1]) * 100
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return model metrics"""
    try:
        return jsonify({
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("BREAST CANCER PREDICTION WEB APPLICATION")
    print("="*60)
    print("\nDISCLAIMER: This system is strictly for educational purposes")
    print("and must not be presented as a medical diagnostic tool.\n")
    
    # Load model artifacts
    if load_model_artifacts():
        print("\nStarting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\n✗ Failed to load model. Please ensure model_development.py has been run.")
