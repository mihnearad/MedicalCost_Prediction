from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Load the model
model = joblib.load('medical_cost_model.pkl')

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract the features from the request
    age = data.get('age')
    gender = data.get('gender')
    bmi = data.get('bmi')
    children = data.get('children')
    smoker = data.get('smoker')
    region = data.get('region')
    
    # Log the received data for debugging
    print(f"Received data: {data}")

    # Create an input array for the model
    input_features = np.array([[age, gender, bmi, children, smoker, region]], dtype=float)
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Log the prediction for debugging
    print(f"Prediction: {prediction}")
    
    # Convert the prediction to a Python float
    prediction = float(prediction)
    
    # Return the prediction as a JSON response
    return jsonify({'predicted_cost': prediction})

if __name__ == '__main__':
    app.run(debug=True)
