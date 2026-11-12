import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load Model Artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
model = joblib.load(os.path.join(MODEL_DIR, 'fertilizer_model.pkl'))
target_encoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))

# Get expected feature names
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = ['N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Moisture']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fertilizer Recommendation System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px auto; max-width: 650px; background-color: #f4f7f6; color: #333; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h2 { color: #2e7d32; text-align: center; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        label { font-weight: bold; font-size: 14px; display: block; margin-bottom: 5px; }
        input { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button { width: 100%; background-color: #2e7d32; color: white; padding: 12px; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background-color: #1b5e20; }
        .result { margin-top: 20px; padding: 15px; background-color: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 4px; text-align: center; font-size: 18px; color: #1b5e20; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🌱 Smart Fertilizer Recommendation System</h2>
        <form method="POST" action="/predict">
            <div class="grid">
                {% for feat in features %}
                <div>
                    <label for="{{ feat }}">{{ feat }}</label>
                    <input type="number" step="any" name="{{ feat }}" required value="50">
                </div>
                {% endfor %}
            </div>
            <button type="submit">Predict Recommended Fertilizer</button>
        </form>

        {% if result %}
        <div class="result">
            <strong>Recommended Fertilizer:</strong> {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, features=feature_names, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = {}
        for feat in feature_names:
            val = request.form.get(feat, 0)
            inputs[feat] = float(val)
            
        input_df = pd.DataFrame([inputs])
        pred_idx = model.predict(input_df)[0]
        recommended = target_encoder.inverse_transform([pred_idx])[0]
        
        return render_template_string(HTML_TEMPLATE, features=feature_names, result=recommended)
    except Exception as e:
        return f"Error making prediction: {str(e)}", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)