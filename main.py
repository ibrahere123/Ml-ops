from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        data.get("Time", 0),
        data.get("V1", 0),
        data.get("V2", 0),
        data.get("V3", 0),
        data.get("V4", 0),
        data.get("V5", 0),
        data.get("V6", 0),
        data.get("V7", 0),
        data.get("V8", 0),
        data.get("V9", 0),
        data.get("V10", 0),
        data.get("V11", 0),
        data.get("V12", 0),
        data.get("V13", 0),
        data.get("V14", 0),
        data.get("V15", 0),
        data.get("V16", 0),
        data.get("V17", 0),
        data.get("V18", 0),
        data.get("V19", 0),
        data.get("V20", 0),
        data.get("V21", 0),
        data.get("V22", 0),
        data.get("V23", 0),
        data.get("V24", 0),
        data.get("V25", 0),
        data.get("V26", 0),
        data.get("V27", 0),
        data.get("V28", 0),
        data.get("Amount", 0)
    ]).reshape(1, -1)

    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
