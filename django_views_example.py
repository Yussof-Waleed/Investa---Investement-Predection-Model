"""
Django Views Integration Example

This file demonstrates how to integrate the Investment Predictor
into your Django application views.

Installation in Django:
1. Copy django_investment_predictor.py to your Django project directory
2. Ensure all requirements are installed: pip install -r requirements.txt
3. Run the Jupyter notebook to train and save models first
4. Import and use in your views as shown below

Example URL patterns (urls.py):
    path('predict/', views.predict_investment_view, name='predict_investment'),
    path('predict/batch/', views.batch_predict_view, name='batch_predict'),
    path('model/status/', views.model_status_view, name='model_status'),
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
import json
import logging

# Import our custom investment predictor
try:
    from django_investment_predictor import (
        InvestmentPredictor,
        django_predict_investment,
        django_predict_investments_batch,
        django_get_model_status
    )
except ImportError:
    # Fallback if module not found
    InvestmentPredictor = None
    django_predict_investment = None
    django_predict_investments_batch = None
    django_get_model_status = None

logger = logging.getLogger(__name__)


# Function-Based Views
@csrf_exempt
@require_http_methods(["POST"])
def predict_investment_view(request):
    """
    API endpoint for single investment prediction.
    
    POST /predict/
    {
        "industry": "Technology",
        "target_market": "B2B",
        "business_model": "SaaS",
        "traction": "Revenue",
        "funding_amount": 1000000,
        "team_size": 8,
        // ... other features
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "success_probability": 0.75,
            "prediction_percentile": "75.0%",
            "risk_level": "Low",
            "risk_emoji": "🟢",
            "recommendation": "Recommended for investment",
            "model_name": "Random Forest",
            "confidence": 0.85
        }
    }
    """
    if not django_predict_investment:
        return JsonResponse({
            'success': False,
            'error': 'Investment predictor not available. Please check installation.'
        }, status=500)
    
    try:
        # Parse JSON data
        data = json.loads(request.body)
        
        # Make prediction
        result = django_predict_investment(data)
        
        # Check for errors in result
        if 'error' in result:
            return JsonResponse({
                'success': False,
                'error': result['error']
            }, status=400)
        
        return JsonResponse({
            'success': True,
            'data': result
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'Internal server error during prediction'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def batch_predict_view(request):
    """
    API endpoint for batch investment predictions.
    
    POST /predict/batch/
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
    """
    if not django_predict_investments_batch:
        return JsonResponse({
            'success': False,
            'error': 'Investment predictor not available'
        }, status=500)
    
    try:
        data = json.loads(request.body)
        investments = data.get('investments', [])
        
        if not investments:
            return JsonResponse({
                'success': False,
                'error': 'No investments provided'
            }, status=400)
        
        # Make batch prediction
        results = django_predict_investments_batch(investments)
        
        return JsonResponse({
            'success': True,
            'data': {
                'predictions': results,
                'total_processed': len(results)
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'Internal server error during batch prediction'
        }, status=500)


@require_http_methods(["GET"])
def model_status_view(request):
    """
    API endpoint to check model status and information.
    
    GET /model/status/
    """
    if not django_get_model_status:
        return JsonResponse({
            'success': False,
            'error': 'Investment predictor not available'
        }, status=500)
    
    try:
        status = django_get_model_status()
        return JsonResponse({
            'success': True,
            'data': status
        })
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'Error checking model status'
        }, status=500)


# Class-Based Views
class InvestmentPredictionView(View):
    """
    Class-based view for investment predictions with form handling.
    """
    
    def get(self, request):
        """
        Render the prediction form.
        """
        context = {
            'industries': [
                'Technology', 'Healthcare', 'Finance', 'E-commerce',
                'Manufacturing', 'Education', 'Real Estate', 'Energy'
            ],
            'target_markets': ['B2B', 'B2C', 'B2B2C'],
            'business_models': [
                'SaaS', 'Marketplace', 'E-commerce', 'Subscription',
                'Freemium', 'Enterprise', 'Consumer'
            ],
            'traction_levels': [
                'Idea', 'Prototype', 'MVP', 'Beta', 'Revenue', 'Growth'
            ]
        }
        return render(request, 'investment_prediction_form.html', context)
    
    def post(self, request):
        """
        Handle form submission and make prediction.
        """
        if not InvestmentPredictor:
            return render(request, 'investment_prediction_form.html', {
                'error': 'Investment predictor not available'
            })
        
        try:
            # Extract data from form
            investment_data = {
                'industry': request.POST.get('industry'),
                'target_market': request.POST.get('target_market'),
                'business_model': request.POST.get('business_model'),
                'traction': request.POST.get('traction'),
                'funding_amount': float(request.POST.get('funding_amount', 0)),
                'time_to_funding': int(request.POST.get('time_to_funding', 12)),
                'team_size': int(request.POST.get('team_size', 5)),
                'advisors_count': int(request.POST.get('advisors_count', 2)),
                'partnerships_count': int(request.POST.get('partnerships_count', 1)),
                'customer_acquisition_cost': float(request.POST.get('customer_acquisition_cost', 100)),
                'lifetime_value': float(request.POST.get('lifetime_value', 1000)),
                'monthly_recurring_revenue': float(request.POST.get('monthly_recurring_revenue', 10000)),
                'churn_rate': float(request.POST.get('churn_rate', 5.0)),
                'market_size': float(request.POST.get('market_size', 1000000000)),
                'competition_level': int(request.POST.get('competition_level', 5)),
                'technology_readiness': int(request.POST.get('technology_readiness', 7)),
                'regulatory_compliance': int(request.POST.get('regulatory_compliance', 8)),
                'intellectual_property': int(request.POST.get('intellectual_property', 5)),
                'geographic_reach': int(request.POST.get('geographic_reach', 3)),
                'revenue_growth_rate': float(request.POST.get('revenue_growth_rate', 15.0)),
                'burn_rate': float(request.POST.get('burn_rate', 50000)),
                'runway_months': int(request.POST.get('runway_months', 18))
            }
            
            # Make prediction
            result = django_predict_investment(investment_data)
            
            if 'error' in result:
                return render(request, 'investment_prediction_form.html', {
                    'error': result['error'],
                    'form_data': investment_data
                })
            
            return render(request, 'investment_prediction_result.html', {
                'result': result,
                'investment_data': investment_data
            })
            
        except Exception as e:
            logger.error(f"Form prediction error: {str(e)}")
            return render(request, 'investment_prediction_form.html', {
                'error': 'Error processing prediction request',
                'form_data': request.POST
            })


# Utility function for integration testing
def test_predictor_integration():
    """
    Test function to verify the predictor works in Django environment.
    Call this from Django shell: python manage.py shell
    >>> from your_app.views import test_predictor_integration
    >>> test_predictor_integration()
    """
    print("🧪 Testing Django Investment Predictor Integration")
    print("=" * 50)
    
    # Test model status
    status = django_get_model_status()
    print(f"Models loaded: {status.get('models_loaded', False)}")
    
    if not status.get('models_loaded'):
        print("❌ Models not loaded. Run the Jupyter notebook first.")
        return False
    
    # Test prediction
    test_data = {
        'industry': 'Technology',
        'target_market': 'B2B',
        'business_model': 'SaaS',
        'traction': 'Revenue',
        'funding_amount': 1000000,
        'team_size': 8
    }
    
    result = django_predict_investment(test_data)
    
    if 'error' in result:
        print(f"❌ Prediction error: {result['error']}")
        return False
    
    print(f"✅ Prediction successful!")
    print(f"Success Probability: {result.get('prediction_percentile', 'N/A')}")
    print(f"Risk Level: {result.get('risk_level', 'N/A')}")
    print(f"Model: {result.get('model_name', 'N/A')}")
    
    return True


# Django REST Framework Views (if using DRF)
try:
    from rest_framework.views import APIView
    from rest_framework.response import Response
    from rest_framework import status
    from rest_framework.decorators import api_view
    
    class InvestmentPredictionAPIView(APIView):
        """
        DRF API View for investment predictions.
        """
        
        def post(self, request):
            """
            Make investment prediction via DRF.
            """
            if not django_predict_investment:
                return Response({
                    'error': 'Investment predictor not available'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            try:
                result = django_predict_investment(request.data)
                
                if 'error' in result:
                    return Response({
                        'error': result['error']
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                return Response(result, status=status.HTTP_200_OK)
                
            except Exception as e:
                return Response({
                    'error': 'Internal server error'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @api_view(['POST'])
    def drf_batch_predict(request):
        """
        DRF function-based view for batch predictions.
        """
        if not django_predict_investments_batch:
            return Response({
                'error': 'Investment predictor not available'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        investments = request.data.get('investments', [])
        if not investments:
            return Response({
                'error': 'No investments provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        results = django_predict_investments_batch(investments)
        
        return Response({
            'predictions': results,
            'total_processed': len(results)
        }, status=status.HTTP_200_OK)

except ImportError:
    # DRF not installed
    InvestmentPredictionAPIView = None
    drf_batch_predict = None


# Example middleware for logging predictions
class InvestmentPredictionLoggingMiddleware:
    """
    Middleware to log all investment predictions for analytics.
    
    Add to MIDDLEWARE in settings.py:
    'your_app.views.InvestmentPredictionLoggingMiddleware'
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        
        # Log prediction requests
        if '/predict/' in request.path and request.method == 'POST':
            try:
                if hasattr(request, 'body'):
                    logger.info(f"Investment prediction request: {request.path}")
                    # Add custom logging logic here
            except Exception as e:
                logger.error(f"Logging error: {str(e)}")
        
        return response 