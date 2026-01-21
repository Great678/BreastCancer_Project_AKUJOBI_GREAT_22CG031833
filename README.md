# Breast Cancer Prediction System

A complete machine learning classification project that predicts whether a tumor is benign or malignant using features from digitized images of breast masses.

**⚠️ DISCLAIMER:** This system is strictly for educational purposes and must not be presented as a medical diagnostic tool.

## Project Overview

This project demonstrates a complete ML classification pipeline with two main components:
- **Part A**: Model Development (model_development.py)
- **Part B**: Web GUI Application (app.py with Flask)

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from scikit-learn, which contains 569 samples of breast mass FNA (Fine Needle Aspirate) measurements.

### Dataset Statistics
- **Total Samples**: 569
  - Benign: 357
  - Malignant: 212
- **Original Features**: 30 (but only 8 recommended)

### Selected Features (5 of 8)
1. **radius_mean** - Average distance from center to perimeter
2. **texture_mean** - Standard deviation of gray-scale values
3. **area_mean** - Mean size of the tumor
4. **smoothness_mean** - Local variation in radius lengths
5. **compactness_mean** - Perimeter² / area - 1.0

### Target Variable
- **diagnosis** - Benign (0) or Malignant (1)

## Algorithm

**Logistic Regression** with the following configuration:
- Solver: lbfgs
- Max iterations: 5000
- Train-test split: 80-20 with stratification

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

Run the model development script to train and save the model:

```bash
python model_development.py
```

This will:
- Load the Breast Cancer Wisconsin dataset from scikit-learn
- Preprocess the data (handle missing values, encode target, scale features)
- Train the Logistic Regression model
- Evaluate using Accuracy, Precision, Recall, F1-Score, and ROC AUC
- Save the model and preprocessing artifacts to the `models/` directory
- Test that the saved model can be reloaded and used for prediction

**Output files created:**
- `models/cancer_model.joblib` - Trained classification model
- `models/scaler.joblib` - Feature scaler
- `models/label_encoder.joblib` - Target variable encoder
- `models/feature_names.joblib` - Feature names
- `models/model_metrics.joblib` - Evaluation metrics

### Step 2: Run the Web Application

Start the Flask web server:

```bash
python app.py
```

The application will start at `http://localhost:5000`

Open your browser and navigate to the URL to access the prediction interface.

## Project Structure

```
Breast_Cancer_Prediction/
├── model_development.py       # Model training script
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/                    # Saved model files
│   ├── cancer_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── feature_names.joblib
│   └── model_metrics.joblib
├── templates/                 # HTML templates
│   └── index.html             # Main prediction interface
└── static/                    # Static files
    ├── styles.css             # CSS styling
    └── script.js              # JavaScript functionality
```

## Model Performance

The model is evaluated on the test set (20% of data) using:

- **Accuracy** - Percentage of correct predictions
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall
- **ROC AUC** - Area under the Receiver Operating Characteristic curve

See console output from `model_development.py` for specific metrics on your model.

## Data Preprocessing

1. **Missing Values Handling:**
   - No missing values present in the dataset
   - Data is clean and ready to use

2. **Feature Encoding:**
   - Target variable (diagnosis) encoded as: 0 = Malignant, 1 = Benign
   - Used LabelEncoder for proper encoding/decoding

3. **Feature Scaling:**
   - All features standardized using StandardScaler (mean=0, std=1)
   - Mandatory for Logistic Regression and distance-based algorithms

## Web Interface Features

- **Input Form**: Easy-to-use form for entering tumor measurements
- **Real-time Validation**: Client-side and server-side input validation
- **Live Predictions**: Instant predictions with confidence scores
- **Probability Visualization**: Shows malignant vs benign probabilities
- **Model Metrics**: Display of model performance metrics
- **Responsive Design**: Works on desktop and mobile devices
- **Educational Disclaimer**: Clear warning about educational use only

## Technical Stack

### Backend
- Python 3.7+
- Flask 2.3.3 - Web framework
- scikit-learn 1.3.0 - ML algorithms
- pandas 2.0.3 - Data manipulation
- numpy 1.24.3 - Numerical computing
- joblib 1.3.1 - Model serialization

### Frontend
- HTML5
- CSS3 (with responsive design)
- Vanilla JavaScript (ES6+)

## API Endpoints

### GET `/`
- Renders the main prediction interface

### POST `/predict`
- **Request body (JSON):**
  ```json
  {
    "radius_mean": 14.5,
    "texture_mean": 19.3,
    "area_mean": 456.2,
    "smoothness_mean": 0.0873,
    "compactness_mean": 0.0846
  }
  ```
- **Response (JSON):**
  ```json
  {
    "success": true,
    "prediction": "Benign",
    "confidence": 95.3,
    "malignant_prob": 4.7,
    "benign_prob": 95.3
  }
  ```

### GET `/metrics`
- Returns model performance metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC)

## Troubleshooting

### Model Not Found
If the web app says "Model not loaded":
1. Ensure you've run `python model_development.py` first
2. Check that the `models/` directory contains all required files

### Port Already in Use
If port 5000 is already in use:
1. Edit `app.py` and change `port=5000` to a different port
2. Or kill the process using port 5000

## Sample Input Values

For testing the application, try these typical values:

**Example 1 (Benign):**
- Radius Mean: 14.5
- Texture Mean: 19.3
- Area Mean: 456.2
- Smoothness Mean: 0.0873
- Compactness Mean: 0.0846

**Example 2 (Malignant):**
- Radius Mean: 20.5
- Texture Mean: 25.3
- Area Mean: 800.0
- Smoothness Mean: 0.1234
- Compactness Mean: 0.2543

## Important Notes

⚠️ **Educational Use Only**: This system is designed for learning purposes in a machine learning course.

❌ **Not a Medical Tool**: This system should NEVER be used for actual medical diagnosis.

📚 **Further Learning**: Users should understand that real medical diagnosis requires professional healthcare providers and additional tests/examinations.

## References

- Dataset: [Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- scikit-learn Documentation: https://scikit-learn.org/
- Flask Documentation: https://flask.palletsprojects.com/
- Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

## License

This project is created for educational purposes as part of CSC415 course project.

## Author

Breast Cancer Prediction System - CSC415 Project - Educational Purposes Only
