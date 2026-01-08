"""
Task 4: Deployment / API (Very Important)
Create a simple API using Flask
Endpoint accepts input data and returns prediction as JSON
"""

from flask import Flask, request, jsonify
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model and preprocessors
print("Loading model and preprocessors...")
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    print("✓ Model and preprocessors loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    label_encoders = None

# Feature configuration
FEATURES = ['Age', 'Gender', 'Tenure_Months', 'Monthly_Charges', 
            'Contract_Type', 'Internet_Service', 'Payment_Method', 'Support_Tickets']
NUMERICAL_FEATURES = ['Age', 'Tenure_Months', 'Monthly_Charges', 'Support_Tickets']

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Check required fields
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            return None, f"Missing fields: {missing}"
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Contract_Type', 'Internet_Service', 'Payment_Method']
        for col in categorical_cols:
            if col in label_encoders:
                value = str(df[col].iloc[0])
                if value in label_encoders[col].classes_:
                    df[col] = label_encoders[col].transform([value])[0]
                else:
                    df[col] = 0  # Unknown category
        
        # Scale numerical features
        df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
        
        return df[FEATURES], None
    
    except Exception as e:
        return None, f"Preprocessing error: {str(e)}"

@app.route('/')
def home():
    """API home page with documentation"""
    return jsonify({
        'message': 'Customer Churn Prediction API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'GET /': 'API documentation',
            'POST /predict': 'Predict churn for a customer',
            'POST /predict_batch': 'Predict churn for multiple customers',
            'GET /health': 'Health check',
            'GET /model_info': 'Model information'
        },
        'example_request': {
            'Age': 35,
            'Gender': 'Male',
            'Tenure_Months': 12,
            'Monthly_Charges': 2500,
            'Contract_Type': 'Month-to-Month',
            'Internet_Service': 'Fiber',
            'Payment_Method': 'Credit Card',
            'Support_Tickets': 3
        }
    })

@app.route('/health')
def health():
    """Check API health"""
    if model is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'message': 'API is ready'
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'message': 'Model not loaded'
        }), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': type(model).__name__,
        'features': FEATURES,
        'n_features': len(FEATURES),
        'output': 'Churn prediction (Yes/No) with probability'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn for a single customer
    
    Expected JSON:
    {
        "Age": 35,
        "Gender": "Male",
        "Tenure_Months": 12,
        "Monthly_Charges": 2500,
        "Contract_Type": "Month-to-Month",
        "Internet_Service": "Fiber",
        "Payment_Method": "Credit Card",
        "Support_Tickets": 3
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess
        processed_data, error = preprocess_input(data)
        if error:
            return jsonify({'error': error}), 400
        
        # Predict
        prediction = model.predict(processed_data)[0]
        proba = model.predict_proba(processed_data)[0]
        
        # Prepare response
        result = {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'churn_probability': float(proba[1]),
            'no_churn_probability': float(proba[0]),
            'confidence': float(max(proba)),
            'risk_level': 'High' if proba[1] > 0.7 else 'Medium' if proba[1] > 0.4 else 'Low',
            'input_data': data
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict churn for multiple customers
    
    Expected JSON:
    {
        "customers": [
            {"Age": 35, "Gender": "Male", ...},
            {"Age": 45, "Gender": "Female", ...}
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'customers' not in data:
            return jsonify({'error': 'No customers data provided'}), 400
        
        customers = data['customers']
        results = []
        
        for idx, customer in enumerate(customers):
            processed_data, error = preprocess_input(customer)
            if error:
                results.append({
                    'customer_index': idx,
                    'error': error,
                    'input_data': customer
                })
                continue
            
            prediction = model.predict(processed_data)[0]
            proba = model.predict_proba(processed_data)[0]
            
            results.append({
                'customer_index': idx,
                'prediction': 'Yes' if prediction == 1 else 'No',
                'churn_probability': float(proba[1]),
                'risk_level': 'High' if proba[1] > 0.7 else 'Medium' if proba[1] > 0.4 else 'Low',
                'input_data': customer
            })
        
        return jsonify({
            'total_customers': len(customers),
            'predictions': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("CUSTOMER CHURN PREDICTION API")
    print("=" * 80)
    print("\nStarting Flask API server...")
    print("API will be available at: http://127.0.0.1:5000")
    print("\nAvailable endpoints:")
    print("  • GET  /            - API documentation")
    print("  • GET  /health      - Health check")
    print("  • GET  /model_info  - Model information")
    print("  • POST /predict     - Single prediction")
    print("  • POST /predict_batch - Batch predictions")
    print("\nPress CTRL+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000)
