from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained Random Forest model
models_dir = os.getenv('MODELS_DIR', 'models')
model_path = os.path.join(models_dir, 'random_forest_model.pkl')

if not os.path.exists(model_path):
    logging.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

rf_model = joblib.load(model_path)
logging.info(f"Model loaded from {model_path}")

# Define the expected number of features
EXPECTED_NUM_FEATURES = 8

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in JSON payload'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        # Check if the number of features matches the expected number
        if features.shape[1] != EXPECTED_NUM_FEATURES:
            return jsonify({'error': f'Expected {EXPECTED_NUM_FEATURES} features, but got {features.shape[1]}'}), 400
        
        prediction = rf_model.predict(features)[0]
        probability = rf_model.predict_proba(features)[0][1]
        return jsonify({'prediction': int(prediction), 'probability': probability})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(debug=True, port=port)