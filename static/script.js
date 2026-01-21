/**
 * Breast Cancer Prediction System - JavaScript
 * Handles form submission, API calls, and UI interactions
 */

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const resultContainer = document.getElementById('resultContainer');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const errorContainer = document.getElementById('errorContainer');
const loadingContainer = document.getElementById('loadingContainer');
const diagnosisResult = document.getElementById('diagnosisResult');
const errorMessageElement = document.getElementById('errorMessage');

// Metric Elements
const accuracyElement = document.getElementById('accuracy');
const precisionElement = document.getElementById('precision');
const recallElement = document.getElementById('recall');
const f1ScoreElement = document.getElementById('f1_score');
const rocAucElement = document.getElementById('roc_auc');

/**
 * Initialize the page
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing...');
    loadMetrics();
});

/**
 * Load model metrics from API
 */
function loadMetrics() {
    fetch('/metrics')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load metrics');
            }
            return response.json();
        })
        .then(data => {
            displayMetrics(data);
        })
        .catch(error => {
            console.error('Error loading metrics:', error);
            // Don't show error to user, just fail silently
        });
}

/**
 * Display metrics on the page
 */
function displayMetrics(metrics) {
    if (metrics.accuracy !== undefined) {
        accuracyElement.textContent = (parseFloat(metrics.accuracy) * 100).toFixed(2) + '%';
    }
    if (metrics.precision !== undefined) {
        precisionElement.textContent = parseFloat(metrics.precision).toFixed(4);
    }
    if (metrics.recall !== undefined) {
        recallElement.textContent = parseFloat(metrics.recall).toFixed(4);
    }
    if (metrics.f1_score !== undefined) {
        f1ScoreElement.textContent = parseFloat(metrics.f1_score).toFixed(4);
    }
    if (metrics.roc_auc !== undefined) {
        rocAucElement.textContent = parseFloat(metrics.roc_auc).toFixed(4);
    }
}

/**
 * Handle form submission
 */
predictionForm.addEventListener('submit', function(e) {
    e.preventDefault();
    submitPrediction();
});

/**
 * Submit prediction request
 */
function submitPrediction() {
    // Get form values
    const radius_mean = document.getElementById('radius_mean').value;
    const texture_mean = document.getElementById('texture_mean').value;
    const area_mean = document.getElementById('area_mean').value;
    const smoothness_mean = document.getElementById('smoothness_mean').value;
    const compactness_mean = document.getElementById('compactness_mean').value;

    // Validate inputs
    if (!validateInputs(radius_mean, texture_mean, area_mean, smoothness_mean, compactness_mean)) {
        return;
    }

    // Show loading state
    showLoading();

    // Prepare request data
    const requestData = {
        radius_mean: parseFloat(radius_mean),
        texture_mean: parseFloat(texture_mean),
        area_mean: parseFloat(area_mean),
        smoothness_mean: parseFloat(smoothness_mean),
        compactness_mean: parseFloat(compactness_mean)
    };

    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Prediction failed');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                displayResult(data);
            } else {
                showError(data.error || 'Prediction failed');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.message || 'An error occurred during prediction. Please try again.');
        });
}

/**
 * Validate form inputs
 */
function validateInputs(radius, texture, area, smoothness, compactness) {
    if (!radius || !texture || !area || !smoothness || !compactness) {
        showError('All fields are required.');
        return false;
    }

    const radiusNum = parseFloat(radius);
    const textureNum = parseFloat(texture);
    const areaNum = parseFloat(area);
    const smoothnessNum = parseFloat(smoothness);
    const compactnessNum = parseFloat(compactness);

    if (radiusNum < 0 || radiusNum > 50) {
        showError('Radius Mean must be between 0 and 50.');
        return false;
    }

    if (textureNum < 0 || textureNum > 50) {
        showError('Texture Mean must be between 0 and 50.');
        return false;
    }

    if (areaNum < 0 || areaNum > 3000) {
        showError('Area Mean must be between 0 and 3000.');
        return false;
    }

    if (smoothnessNum < 0 || smoothnessNum > 1) {
        showError('Smoothness Mean must be between 0 and 1.');
        return false;
    }

    if (compactnessNum < 0 || compactnessNum > 1) {
        showError('Compactness Mean must be between 0 and 1.');
        return false;
    }

    return true;
}

/**
 * Display prediction result
 */
function displayResult(data) {
    hideLoading();
    hideError();
    
    // Set diagnosis
    diagnosisResult.textContent = data.prediction;
    
    // Update result card styling
    resultCard.classList.remove('benign', 'malignant');
    if (data.prediction === 'Benign') {
        resultCard.classList.add('benign');
        resultIcon.textContent = '🟢';
    } else {
        resultCard.classList.add('malignant');
        resultIcon.textContent = '🔴';
    }
    
    // Set confidence
    document.getElementById('confidence').textContent = data.confidence.toFixed(2);
    
    // Set probability bars
    const malignantBar = document.getElementById('malignantBar');
    const benignBar = document.getElementById('benignBar');
    const malignantProb = document.getElementById('malignantProb');
    const benignProb = document.getElementById('benignProb');
    
    malignantBar.style.width = data.malignant_prob.toFixed(1) + '%';
    benignBar.style.width = data.benign_prob.toFixed(1) + '%';
    malignantProb.textContent = data.malignant_prob.toFixed(2) + '%';
    benignProb.textContent = data.benign_prob.toFixed(2) + '%';
    
    // Show result
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Show error message
 */
function showError(message) {
    hideLoading();
    errorMessageElement.textContent = message;
    errorContainer.classList.remove('hidden');
    resultContainer.classList.add('hidden');
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide error message
 */
function hideError() {
    errorContainer.classList.add('hidden');
}

/**
 * Show loading state
 */
function showLoading() {
    hideError();
    resultContainer.classList.add('hidden');
    loadingContainer.classList.remove('hidden');
}

/**
 * Hide loading state
 */
function hideLoading() {
    loadingContainer.classList.add('hidden');
}
