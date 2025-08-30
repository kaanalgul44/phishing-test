from flask import Flask, request, jsonify
from flask_cors import CORS
from production_system import ProductionPhishingSystem

app = Flask(__name__)
CORS(app)
system = ProductionPhishingSystem()
system.load_model()

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.json.get('email_text')
    result = system.detector.predict_email(email_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)