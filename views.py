import os
import pandas as pd
import joblib
from datetime import datetime
from firebase_admin import db
from rest_framework.decorators import api_view
from rest_framework.response import Response


def load_model_components():
    """Load the trained model and preprocessing components"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'saved_models')
    
    return {
        'model': joblib.load(os.path.join(model_dir, 'best_model_random_forest.pkl')),
        'scaler': joblib.load(os.path.join(model_dir, 'feature_scaler.pkl')),
        'encoders': joblib.load(os.path.join(model_dir, 'label_encoders.pkl')),
        'feature_names': joblib.load(os.path.join(model_dir, 'feature_names.pkl')),
        'categorical_columns': joblib.load(os.path.join(model_dir, 'categorical_columns.pkl'))
    }


def prepare_prediction_data(project_info, analysis_info):
    """Transform Firebase project data into model-compatible format"""
    # Handle potential missing or invalid numeric values
    def safe_float(value, default=0.0):
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default
    
    def safe_int(value, default=0):
        try:
            return int(value) if value else default
        except (ValueError, TypeError):
            return default
    
    # Calculate profit margin safely
    revenue = safe_float(project_info.get('annualRevenue'))
    profit = safe_float(project_info.get('netProfit'))
    profit_margin = profit / revenue if revenue > 0 else 0.0
    
    # USD conversion rate (approximate)
    usd_rate = 50.0
    
    return {
        'industry': project_info.get('projectCategory', 'Other'),
        'funding_egp': safe_float(project_info.get('fundingNeeded')),
        'equity_percentage': safe_float(project_info.get('ownershipPercentage')),
        'duration_months': 12,  # Default value
        'target_market': 'Regional',  # Default based on location
        'business_model': analysis_info.get('businessModel', 'B2C'),
        'founder_experience_years': 5,  # Default value
        'team_size': safe_int(project_info.get('teamSize', 1)),
        'traction': 'Medium',  # Default value
        'market_size_usd': 5000000,  # Default value
        'funding_usd': safe_float(project_info.get('fundingNeeded')) / usd_rate,
        'profit': profit,
        'repeat_purchase_rate': safe_float(project_info.get('repeatPurchaseRate')),
        'branches_count': safe_int(project_info.get('numberOfBranches')),
        'revenue': revenue,
        'customers': safe_int(project_info.get('currentCustomers')),
        'revenue_growth': safe_float(project_info.get('monthlyGrowthRate')),
        'profit_margin': profit_margin,
        'customer_growth': safe_float(project_info.get('customerGrowthRate')),
        'churn_rate': safe_float(project_info.get('churnRate')),
        'operating_costs': safe_float(project_info.get('monthlyOperatingCosts')),
        'debt_to_profit_ratio': safe_float(project_info.get('debtToEquityRatio'))
    }


def predict_investment_success(data_df, components):
    """Make predictions using the loaded model and preprocessors"""
    data_processed = data_df.copy()
    
    # Encode categorical variables
    for col in components['categorical_columns']:
        if col in data_processed.columns:
            encoder = components['encoders'][col]
            data_processed[col] = data_processed[col].astype(str)
            
            # Handle unseen categories
            def safe_transform(value):
                try:
                    return encoder.transform([value])[0]
                except ValueError:
                    most_frequent_class = encoder.classes_[0]
                    return encoder.transform([most_frequent_class])[0]
            
            data_processed[col] = data_processed[col].apply(safe_transform)
    
    # Ensure all required features are present
    for feature in components['feature_names']:
        if feature not in data_processed.columns:
            data_processed[feature] = 0
    
    # Select features in correct order
    data_processed = data_processed[components['feature_names']]
    
    # Scale features if needed (Random Forest typically doesn't need scaling)
    if type(components['model']).__name__ in ['LinearRegression', 'Ridge', 'Lasso', 'SVR', 'MLPRegressor']:
        data_processed = components['scaler'].transform(data_processed)
    else:
        data_processed = data_processed.values
    
    # Make prediction
    return components['model'].predict(data_processed)[0]


@api_view(['GET'])
def predict_investment(request, project_id):
    """API endpoint to get investment success prediction for a project"""
    try:
        # Load model components
        components = load_model_components()
        
        # Fetch project data from Firebase
        project_info = db.reference(f'projects/{project_id}').get()
        analysis_info = db.reference(f'analysis/{project_id}').get()
        
        if not project_info or not analysis_info:
            return Response({"error": "Project not found"}, status=404)
        
        # Prepare data for prediction
        prediction_data = prepare_prediction_data(project_info, analysis_info)
        df = pd.DataFrame([prediction_data])
        
        # Make prediction
        success_probability = predict_investment_success(df, components)
        
        # Determine risk level and recommendation
        if success_probability < 0.3:
            risk_level = "High Risk"
            recommendation = "Not Recommended"
            risk_emoji = "🔴"
        elif success_probability < 0.7:
            risk_level = "Medium Risk"
            recommendation = "Consider with Caution"
            risk_emoji = "🟡"
        else:
            risk_level = "Low Risk"
            recommendation = "Recommended Investment"
            risk_emoji = "🟢"
        
        # Prepare key insights based on data
        key_insights = []
        
        # Revenue insights
        revenue = prediction_data['revenue']
        if revenue > 0:
            if revenue > 1000000:
                key_insights.append("Strong revenue performance")
            elif revenue < 100000:
                key_insights.append("Revenue is below market average")
        
        # Growth insights
        if prediction_data['revenue_growth'] > 0.25:
            key_insights.append("Excellent growth trajectory")
        elif prediction_data['revenue_growth'] < 0.05:
            key_insights.append("Growth rate is concerning")
        
        # Customer insights
        if prediction_data['repeat_purchase_rate'] > 0.7:
            key_insights.append("Strong customer retention")
        elif prediction_data['churn_rate'] > 0.3:
            key_insights.append("High churn rate is a risk factor")
        
        # Financial health
        if prediction_data['profit_margin'] > 0.2:
            key_insights.append("Healthy profit margins")
        elif prediction_data['profit_margin'] < 0:
            key_insights.append("Currently operating at a loss")
        
        # Return comprehensive response
        response_data = {
            "project_id": project_id,
            "project_name": project_info.get('projectName', 'Unnamed Project'),
            "prediction": {
                "success_probability": float(success_probability),
                "success_percentage": float(success_probability * 100),
                "risk_level": risk_level,
                "risk_indicator": risk_emoji,
                "recommendation": recommendation
            },
            "key_insights": key_insights,
            "analysis_timestamp": datetime.now().isoformat(),
            "model_info": "Random Forest Regressor (R² score: 0.87)"
        }
        
        return Response(response_data)
        
    except Exception as e:
        return Response({
            "error": "Failed to generate prediction",
            "details": str(e)
        }, status=500)
    

# Example usage:
# response = predict_investment(request, 'project123')
# print(response.data)


# Example urlpatterns = [
#     path('api/predict_investment/<str:project_id>/', predict_investment),
# ]