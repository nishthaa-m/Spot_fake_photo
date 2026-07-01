import os
import base64
import tempfile
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add current directory to path to ensure predict is importable
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_DIR)
from predict import predict

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    # Serves the index.html file from static/
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
            
        # Parse base64 data URL (e.g. data:image/jpeg;base64,...)
        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',')[1]
            
        img_bytes = base64.b64decode(img_data)
        
        # Write bytes to a temporary file to pass to the predictor
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(img_bytes)
            temp_path = temp_file.name
            
        try:
            # Call our prediction logic
            score = predict(temp_path)
        finally:
            # Always ensure the temp file is deleted
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return jsonify({
            "success": True,
            "score": score,
            "label": "screen" if score >= 0.5 else "real"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Spot the Fake Photo Live Demo Server...")
    print("Open http://localhost:5000 in your browser to view.")
    app.run(host='0.0.0.0', port=5000, debug=True)
