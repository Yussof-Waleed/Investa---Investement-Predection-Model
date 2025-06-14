# Investa Prediction Model Handover Documentation

## 1. Project Overview

This documentation provides a comprehensive guide to the Investa Prediction Model, including the model training process, API implementation, and integration guidelines. The model predicts investment success probability based on project and business analysis data.

## 2. Requirements and Setup

### 2.1 Dependencies

```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==1.7.6
joblib==1.3.2
scipy==1.11.1
jupyter==1.0.0
notebook==7.0.2
```

### 2.2 Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the saved_models directory exists with model artifacts

## 3. Model Training Process

The model was trained using the Jupyter notebook investment_prediction_models.ipynb. The notebook contains:

1. **Data Loading and Exploration** - Analysis of the dataset structure and features
2. **Data Preprocessing** - Encoding categorical variables and scaling numerical features
3. **Model Training and Comparison** - Training and evaluation of 9 different algorithms
4. **Model Selection** - Selection of the best performing model (Random Forest Regressor)
5. **Feature Importance Analysis** - Identification of most predictive features
6. **Model Saving** - Saving model artifacts for production use

### 3.1 Training Results Summary

- **Best Model**: Random Forest Regressor
- **Performance Metrics**:
  - R² Score: ~0.87 (explains 87% of variance)
  - RMSE: Low error values indicate good fit
  - MAE: Provides average prediction error magnitude
- **Key Predictive Features**: Profit margin, revenue, customer growth, etc.

## 4. Model Files and Artifacts

After training, the following files are generated in the saved_models directory:

| File | Description |
|------|-------------|
| `best_model_random_forest.pkl` | Trained Random Forest model |
| `feature_scaler.pkl` | StandardScaler for numerical features |
| `label_encoders.pkl` | Dictionary of LabelEncoders for categorical features |
| `feature_names.pkl` | List of feature names in correct order |
| `categorical_columns.pkl` | List of categorical column names |
| `model_metrics.pkl` | Performance metrics and model information |

## 5. API Integration

### 5.1 API Endpoint

The API endpoint is defined in views.py:

```python
@api_view(['GET'])
def predict_investment(request, project_id):
    """API endpoint to get investment success prediction for a project"""
    # Implementation details in views.py
```

#### Request Format
- **Method**: GET
- **URL**: `/api/predict_investment/<project_id>/`
- **Parameters**: `project_id` - ID of project in Firebase

#### Response Format
```json
{
  "project_id": "project123",
  "project_name": "Project Name",
  "prediction": {
    "success_probability": 0.78,
    "success_percentage": 78.0,
    "risk_level": "Low Risk",
    "risk_indicator": "🟢",
    "recommendation": "Recommended Investment"
  },
  "key_insights": [
    "Strong revenue performance",
    "Excellent growth trajectory"
  ],
  "analysis_timestamp": "2025-06-14T10:30:45.123456",
  "model_info": "Random Forest Regressor (R² score: 0.87)"
}
```

### 5.2 Utility Functions

#### `load_model_components()`
Loads all saved model artifacts from the saved_models directory.

#### `prepare_prediction_data(project_info, analysis_info)`
Transforms Firebase project data into model-compatible format with safe value handling.

#### `predict_investment_success(data_df, components)`
Makes predictions using the loaded model and preprocessors with robustness to unseen categories.

## 6. Safe Integration Guidelines

### 6.1 Error Handling Practices

The API implementation includes:
- Robust error handling for Firebase data retrieval
- Safe type conversion with default values
- Handling of unseen categorical values
- Comprehensive try-except blocks
- Informative error responses

### 6.2 Integration Steps

1. **URL Configuration**:
   ```python
   urlpatterns = [
       path('api/predict_investment/<str:project_id>/', predict_investment),
   ]
   ```

2. **Firebase Setup**:
   - Ensure Firebase Admin SDK is initialized
   - Verify database structure matches expected format
   - Test connection before deployment

3. **Model Files Installation**:
   - Copy all files from saved_models directory to the production server
   - Maintain the file structure referenced in `load_model_components()`

### 6.3 Security Considerations

1. **API Authentication** - Implement authentication to protect the endpoint
2. **Input Validation** - Validate project IDs before accessing Firebase
3. **Rate Limiting** - Implement rate limiting to prevent abuse
4. **Monitoring** - Add logging to track usage and errors

## 7. Testing and Validation

### 7.1 Testing the API

1. **Unit Tests**:
   ```python
   # Example test case
   def test_predict_investment():
       response = client.get('/api/predict_investment/test_project_id/')
       assert response.status_code == 200
       assert 'prediction' in response.data
       assert 'success_probability' in response.data['prediction']
   ```

2. **Integration Tests**:
   - Test with various Firebase project data scenarios
   - Verify error handling for missing/malformed data

### 7.2 Validation Approaches

1. **Compare predictions with manual assessments**
2. **Monitor prediction distributions for drift**
3. **Track actual investment outcomes to validate model**

## 8. Troubleshooting and FAQs

### 8.1 Common Issues

1. **Model File Not Found**:
   - Verify paths in `load_model_components()`
   - Check file permissions and existence

2. **Firebase Connection Issues**:
   - Verify Firebase credentials and initialization
   - Check network connectivity and security rules

3. **Unexpected Predictions**:
   - Review input data for anomalies
   - Verify feature transformations match training process

### 8.2 Model Retraining Guidelines

The model should be retrained:
- When new significant data becomes available
- If prediction performance degrades
- At regular intervals (e.g., quarterly)

Retrain using the notebook investment_prediction_models.ipynb and update model artifacts.

---
