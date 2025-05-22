from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the XGBoost model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')

        if not features or len(features) != model.n_features_in_:
            return jsonify({'error': f'Invalid number of features. Expected {model.n_features_in_}'}), 400

        # Preprocess input
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
