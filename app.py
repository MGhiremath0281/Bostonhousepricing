from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create the Flask app
app = Flask(__name__)

# Load your trained model and scaler (ensure they are both saved during training)
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))  # Load the scaler (if it was used during training)

@app.route('/')
def home():
    return 'Welcome to the Flask App!'

# Route for the predict API (handles POST requests)
@app.route('/pridict_api', methods=['POST'])
def predict_api():
    try:
        # Get the input data from the request body (as JSON)
        data = request.get_json()

        # Validate the input data
        if 'input' not in data:
            return jsonify({"error": "No input data provided"}), 400

        input_data = data['input']

        # Check if input_data has exactly 8 features
        if len(input_data) != 8:
            return jsonify({"error": "Input must have exactly 8 features."}), 400
        
        # Convert the input data into a numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Preprocess the input data using the same scaler as during training
        scaled_input = scaler.transform(input_array)

        # Make the prediction using the scaled input
        prediction = model.predict(scaled_input)[0]

        # Post-processing: Ensure that the prediction is non-negative
        if prediction < 0:
            prediction = 0

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction})

    except Exception as e:
        # Handle any exceptions and return an error message
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Runs the app in debug mode
