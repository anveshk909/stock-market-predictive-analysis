from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained Random Forest model
models_dir = 'models'
rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))

# Define the expected number of features
EXPECTED_NUM_FEATURES = 8

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    
    # Check if the number of features matches the expected number
    if features.shape[1] != EXPECTED_NUM_FEATURES:
        return jsonify({'error': f'Expected {EXPECTED_NUM_FEATURES} features, but got {features.shape[1]}'}), 400
    
    prediction = rf_model.predict(features)[0]
    probability = rf_model.predict_proba(features)[0][1]
    return jsonify({'prediction': int(prediction), 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True, port=5001)