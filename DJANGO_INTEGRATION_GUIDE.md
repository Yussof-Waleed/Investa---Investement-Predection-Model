# Django Integration Guide for Investment Predictor

This guide shows you how to integrate the Investment Prediction ML models into your Django application.

## 📋 Prerequisites

1. **Train the Models First**
   ```bash
   # Run the Jupyter notebook to train and save models
   jupyter notebook investment_prediction_models.ipynb
   # Run all cells to completion
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Model Files**
   Ensure the `saved_models/` directory exists with these files:
   - `best_model_*.pkl`
   - `feature_scaler.pkl`
   - `label_encoders.pkl`
   - `feature_names.pkl`
   - `categorical_columns.pkl`
   - `model_metrics.pkl`

## 🚀 Quick Start

### 1. Copy Files to Django Project

```bash
# Copy the prediction module to your Django project
cp django_investment_predictor.py /path/to/your/django/project/
cp django_views_example.py /path/to/your/django/project/your_app/views.py

# Copy model directory
cp -r saved_models/ /path/to/your/django/project/
```

### 2. Add to Django URLs

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    # API endpoints
    path('api/predict/', views.predict_investment_view, name='predict_investment'),
    path('api/predict/batch/', views.batch_predict_view, name='batch_predict'),
    path('api/model/status/', views.model_status_view, name='model_status'),
    
    # Web interface
    path('predict/', views.InvestmentPredictionView.as_view(), name='prediction_form'),
]
```

### 3. Basic Usage in Views

```python
# views.py
from django_investment_predictor import django_predict_investment

def my_prediction_view(request):
    investment_data = {
        'industry': 'Technology',
        'target_market': 'B2B',
        'business_model': 'SaaS',
        'traction': 'Revenue',
        'funding_amount': 1000000,
        'team_size': 8,
        # ... other features
    }
    
    result = django_predict_investment(investment_data)
    
    if 'error' in result:
        # Handle error
        return JsonResponse({'error': result['error']})
    
    return JsonResponse({
        'success_probability': result['success_probability'],
        'risk_level': result['risk_level'],
        'recommendation': result['recommendation']
    })
```

## 📊 API Reference

### Single Prediction

**Endpoint:** `POST /api/predict/`

**Request Body:**
```json
{
    "industry": "Technology",
    "target_market": "B2B",
    "business_model": "SaaS",
    "traction": "Revenue",
    "funding_amount": 1000000,
    "time_to_funding": 6,
    "team_size": 8,
    "advisors_count": 3,
    "partnerships_count": 2,
    "customer_acquisition_cost": 150,
    "lifetime_value": 2000,
    "monthly_recurring_revenue": 50000,
    "churn_rate": 3.0,
    "market_size": 5000000000,
    "competition_level": 6,
    "technology_readiness": 8,
    "regulatory_compliance": 9,
    "intellectual_property": 7,
    "geographic_reach": 4,
    "revenue_growth_rate": 25.0,
    "burn_rate": 75000,
    "runway_months": 24
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "success_probability": 0.75,
        "prediction_percentile": "75.0%",
        "risk_level": "Low",
        "risk_emoji": "🟢",
        "recommendation": "Recommended for investment",
        "model_name": "Random Forest",
        "confidence": 0.85,
        "input_data": { /* validated input data */ }
    }
}
```

### Batch Predictions

**Endpoint:** `POST /api/predict/batch/`

**Request Body:**
```json
{
    "investments": [
        {
            "industry": "Technology",
            "target_market": "B2B",
            // ... investment 1 data
        },
        {
            "industry": "Healthcare",
            "target_market": "B2C",
            // ... investment 2 data
        }
    ]
}
```

### Model Status

**Endpoint:** `GET /api/model/status/`

**Response:**
```json
{
    "success": true,
    "data": {
        "models_loaded": true,
        "model_info": {
            "model_name": "Random Forest",
            "test_metrics": { /* performance metrics */ },
            "feature_names": [ /* list of features */ ]
        }
    }
}
```

## 🔧 Advanced Integration

### Custom Model Directory

```python
from django_investment_predictor import InvestmentPredictor

# Use custom model directory
predictor = InvestmentPredictor(models_dir="/path/to/models")
result = predictor.predict_single(investment_data)
```

### Error Handling

```python
from django_investment_predictor import InvestmentPredictorError

try:
    result = django_predict_investment(data)
except InvestmentPredictorError as e:
    # Handle prediction-specific errors
    logger.error(f"Prediction failed: {e}")
    return JsonResponse({'error': str(e)}, status=400)
except Exception as e:
    # Handle general errors
    logger.error(f"Unexpected error: {e}")
    return JsonResponse({'error': 'Internal error'}, status=500)
```

### Feature Importance

```python
from django_investment_predictor import InvestmentPredictor

predictor = InvestmentPredictor()
importance = predictor.get_feature_importance(top_n=10)

# Returns: {'feature_name': importance_score, ...}
```

## 🧪 Testing Integration

### Django Shell Testing

```python
# python manage.py shell
from django_investment_predictor import django_get_model_status, django_predict_investment

# Check model status
status = django_get_model_status()
print(f"Models loaded: {status['models_loaded']}")

# Test prediction
test_data = {
    'industry': 'Technology',
    'target_market': 'B2B',
    'business_model': 'SaaS',
    'traction': 'Revenue',
    'funding_amount': 1000000
}

result = django_predict_investment(test_data)
print(f"Success probability: {result['prediction_percentile']}")
```

### Unit Tests

```python
# tests.py
from django.test import TestCase
from django_investment_predictor import django_predict_investment

class InvestmentPredictorTests(TestCase):
    
    def test_valid_prediction(self):
        data = {
            'industry': 'Technology',
            'target_market': 'B2B',
            'business_model': 'SaaS',
            'traction': 'Revenue',
            'funding_amount': 1000000
        }
        
        result = django_predict_investment(data)
        
        self.assertNotIn('error', result)
        self.assertIn('success_probability', result)
        self.assertIsInstance(result['success_probability'], float)
        self.assertGreaterEqual(result['success_probability'], 0)
        self.assertLessEqual(result['success_probability'], 1)
    
    def test_missing_required_fields(self):
        data = {'funding_amount': 1000000}  # Missing required fields
        
        result = django_predict_investment(data)
        
        self.assertIn('error', result)
```

## 🔐 Production Considerations

### Security

1. **Input Validation**
   ```python
   def validate_investment_data(data):
       required_fields = ['industry', 'target_market', 'business_model', 'traction']
       for field in required_fields:
           if field not in data:
               raise ValueError(f"Missing required field: {field}")
       
       # Additional validation logic
       if data.get('funding_amount', 0) < 0:
           raise ValueError("Funding amount must be positive")
   ```

2. **Rate Limiting**
   ```python
   # Use django-ratelimit
   from django_ratelimit.decorators import ratelimit
   
   @ratelimit(key='ip', rate='10/m', method='POST')
   def predict_investment_view(request):
       # Your prediction logic
   ```

### Performance

1. **Model Caching**
   ```python
   # settings.py
   CACHES = {
       'default': {
           'BACKEND': 'django.core.cache.backends.redis.RedisCache',
           'LOCATION': 'redis://127.0.0.1:6379/1',
       }
   }
   
   # views.py
   from django.core.cache import cache
   
   def get_cached_predictor():
       predictor = cache.get('investment_predictor')
       if predictor is None:
           predictor = InvestmentPredictor()
           cache.set('investment_predictor', predictor, 3600)  # 1 hour
       return predictor
   ```

2. **Async Views** (Django 3.1+)
   ```python
   import asyncio
   from django.http import JsonResponse
   from asgiref.sync import sync_to_async
   
   async def async_predict_view(request):
       data = json.loads(request.body)
       
       # Run prediction in thread pool
       result = await sync_to_async(django_predict_investment)(data)
       
       return JsonResponse({'data': result})
   ```

### Monitoring

1. **Logging**
   ```python
   # settings.py
   LOGGING = {
       'version': 1,
       'handlers': {
           'file': {
               'level': 'INFO',
               'class': 'logging.FileHandler',
               'filename': 'investment_predictions.log',
           },
       },
       'loggers': {
           'django_investment_predictor': {
               'handlers': ['file'],
               'level': 'INFO',
           },
       },
   }
   ```

2. **Metrics Collection**
   ```python
   from django.db import models
   
   class PredictionLog(models.Model):
       timestamp = models.DateTimeField(auto_now_add=True)
       industry = models.CharField(max_length=100)
       funding_amount = models.FloatField()
       success_probability = models.FloatField()
       model_name = models.CharField(max_length=100)
       processing_time = models.FloatField()  # in seconds
   ```

## 📱 Frontend Integration

### JavaScript Example

```javascript
// Make prediction request
async function predictInvestment(investmentData) {
    try {
        const response = await fetch('/api/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify(investmentData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPrediction(result.data);
        } else {
            displayError(result.error);
        }
    } catch (error) {
        console.error('Prediction error:', error);
        displayError('Network error occurred');
    }
}

function displayPrediction(data) {
    document.getElementById('probability').textContent = data.prediction_percentile;
    document.getElementById('risk-level').textContent = data.risk_level;
    document.getElementById('recommendation').textContent = data.recommendation;
}
```

### HTML Form Example

```html
<!-- investment_prediction_form.html -->
<form method="post" id="prediction-form">
    {% csrf_token %}
    
    <div class="form-group">
        <label for="industry">Industry:</label>
        <select name="industry" id="industry" required>
            <option value="">Select Industry</option>
            <option value="Technology">Technology</option>
            <option value="Healthcare">Healthcare</option>
            <option value="Finance">Finance</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="funding_amount">Funding Amount ($):</label>
        <input type="number" name="funding_amount" id="funding_amount" required>
    </div>
    
    <!-- Add more fields as needed -->
    
    <button type="submit">Predict Investment Success</button>
</form>
```

## 🛠️ Troubleshooting

### Common Issues

1. **Models Not Loading**
   - Ensure the Jupyter notebook has been run completely
   - Check that `saved_models/` directory exists and contains all required files
   - Verify file permissions

2. **Import Errors**
   - Install all required packages: `pip install -r requirements.txt`
   - Check Python path and module locations

3. **Prediction Errors**
   - Validate input data format
   - Check for missing required categorical fields
   - Ensure numerical values are valid

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('django_investment_predictor').setLevel(logging.DEBUG)

# Test model loading
from django_investment_predictor import InvestmentPredictor
predictor = InvestmentPredictor()
print(f"Models loaded: {predictor.is_loaded}")
print(f"Model info: {predictor.get_model_info()}")
```

## 📞 Support

For issues or questions:
1. Check the model training logs in the Jupyter notebook
2. Verify all dependencies are installed correctly
3. Test the prediction script independently before Django integration
4. Check Django logs for detailed error messages

## 🔄 Updates and Maintenance

### Retraining Models

```python
# When you retrain models, Django will automatically pick up new model files
# No code changes needed if model structure remains the same

# To force model reload:
from django_investment_predictor import InvestmentPredictor
predictor = InvestmentPredictor()
predictor.load_models()  # Reload from disk
```

### Model Versioning

Consider implementing model versioning for production:

```python
# Use timestamped model directories
models_dir = f"saved_models_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
predictor = InvestmentPredictor(models_dir=models_dir)
``` 