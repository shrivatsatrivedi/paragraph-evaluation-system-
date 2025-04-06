from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load trained model
model = joblib.load("essay_grader.pkl")

@app.route('/')
def home():
    return render_template("index.html")  # Serve frontend

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    essay = data.get('essay', '')

    if not essay:
        return jsonify({'error': 'No essay provided'}), 400

    # Predict score
    score = model.predict([essay])[0]
    return jsonify({'predicted_score': round(score, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
