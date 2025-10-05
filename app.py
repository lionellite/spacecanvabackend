from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Import workspace routes
from workspace_routes import workspace_bp
app.register_blueprint(workspace_bp)

# Import proxy routes
from proxy_routes import proxy_bp
app.register_blueprint(proxy_bp)

# Load the Keras model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'exo_full.keras')
model = None

# Constants for curve generation
CURVE_LENGTH = 1000  # Adjust based on your model's expected curve length

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"⚠️ Model file not found at {MODEL_PATH}")
            print("Please place your exoplanet_model.keras file in the backend/model/ directory")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def generate_transit_curve(period, duration, depth):
    """
    Generate a synthetic transit light curve from period, duration, and depth.
    This is a simplified model - adjust based on your training data generation.
    
    Args:
        period: Orbital period in days
        duration: Transit duration in hours
        depth: Transit depth in ppm
        
    Returns:
        numpy array of shape (CURVE_LENGTH,) representing the light curve
    """
    # Convert depth from ppm to fraction
    depth_fraction = depth / 1e6
    
    # Create time array
    time = np.linspace(-0.5, 0.5, CURVE_LENGTH)
    
    # Calculate transit parameters
    duration_phase = (duration / 24.0) / period  # Convert hours to phase units
    
    # Generate simple box-shaped transit
    curve = np.ones(CURVE_LENGTH)
    
    # Apply transit dip
    transit_mask = np.abs(time) < (duration_phase / 2)
    curve[transit_mask] = 1.0 - depth_fraction
    
    # Add smooth edges (ingress/egress)
    edge_width = int(CURVE_LENGTH * 0.02)  # 2% of curve for edges
    for i in range(len(curve)):
        if transit_mask[i]:
            # Find distance to edge
            dist_to_edge = min(
                abs(i - np.where(transit_mask)[0][0]),
                abs(i - np.where(transit_mask)[0][-1])
            )
            if dist_to_edge < edge_width:
                # Smooth transition
                smooth_factor = dist_to_edge / edge_width
                curve[i] = 1.0 - depth_fraction * smooth_factor
    
    return curve

# Load model on startup (but don't fail if not found)
load_model()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

def parse_malformed_query_string(query_string):
    """Parse malformed query string with quoted keys and values."""
    if not query_string:
        return None

    # Handle malformed format like ?"mission"=%20"kepler"&"period"=%203.52
    # Remove leading ? and split by &
    params = query_string.lstrip('?').split('&')
    data = {}
    for param in params:
        if '=' in param:
            key, value = param.split('=', 1)
            # Remove surrounding quotes if present
            key = key.strip('"')
            value = value.strip('"')
            # URL decode the value
            from urllib.parse import unquote
            value = unquote(value)
            data[key] = value
    return data

def predict_exoplanet_internal(data):
    """
    Internal function to make predictions
    Can be called from routes or other functions
    """
    if model is None:
        raise Exception('Model not loaded')
    
    # Extract transit parameters for curve generation
    period = float(data.get('period', 0))
    duration = float(data.get('duration', 0))
    depth = float(data.get('depth', 0))
    
    # Generate curve_input from transit parameters
    curve_input = generate_transit_curve(period, duration, depth)
    curve_input = curve_input.reshape(1, CURVE_LENGTH, 1)
    
    # Extract feat_input
    feat_input = np.array([
        float(data.get('impact', 0)),
        float(data.get('snr', 0)),
        float(data.get('steff', 0)),
        float(data.get('srad', 0)),
        float(data.get('slogg', 0)),
        float(data.get('tmag', 0))
    ]).reshape(1, -1)
    
    # Make prediction
    predictions = model.predict([curve_input, feat_input], verbose=0)
    
    # Extract outputs
    label_probs = predictions[0][0]
    period_pred = float(predictions[1][0][0])
    depth_pred = float(predictions[2][0][0])
    
    # Determine classification
    label_classes = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
    predicted_label_idx = np.argmax(label_probs)
    predicted_label = label_classes[predicted_label_idx]
    confidence = float(label_probs[predicted_label_idx])
    
    # Map to classification
    is_exoplanet = predicted_label in ['CANDIDATE', 'CONFIRMED']
    if predicted_label == 'CONFIRMED':
        classification = "Confirmed Exoplanet"
    elif predicted_label == 'CANDIDATE':
        classification = "Candidate Exoplanet"
    else:
        classification = "False Positive"
    
    # Calculate metrics
    planet_radius = None
    if data.get('depth') and data.get('srad'):
        planet_radius = float(data.get('srad')) * np.sqrt(float(data.get('depth')) / 1e6)
    
    depth_pred_ppm = depth_pred * 1e6
    
    return {
        'prediction': {
            'is_exoplanet': is_exoplanet,
            'label': predicted_label,
            'label_probabilities': {
                'CANDIDATE': float(label_probs[0]),
                'CONFIRMED': float(label_probs[1]),
                'FALSE_POSITIVE': float(label_probs[2])
            },
            'confidence': confidence,
            'classification': classification
        },
        'model_predictions': {
            'predicted_period': period_pred,
            'predicted_depth_ppm': depth_pred_ppm,
            'input_period': period,
            'input_depth_ppm': depth,
            'period_error': abs(period_pred - period),
            'depth_error_ppm': abs(depth_pred_ppm - depth)
        },
        'metadata': {
            'mission': data.get('mission', 'unknown'),
            'period': period,
            'duration': duration,
            'depth': depth,
            'impact': data.get('impact'),
            'snr': data.get('snr'),
            'steff': data.get('steff'),
            'srad': data.get('srad'),
            'slogg': data.get('slogg'),
            'tmag': data.get('tmag'),
            'estimated_planet_radius': planet_radius
        },
        'analysis': {
            'transit_quality': 'good' if data.get('snr', 0) > 10 else 'poor',
            'transit_type': 'central' if data.get('impact', 1) < 0.7 else 'grazing',
            'star_type': 'main_sequence' if 4.0 <= data.get('slogg', 0) <= 5.0 else 'evolved',
            'model_consistency': 'high' if abs(period_pred - period) < period * 0.1 and abs(depth_pred_ppm - depth) < depth * 0.2 else 'low'
        }
    }

@app.route('/api/predict', methods=['POST'])
def predict_exoplanet():
    """
    Predict if a candidate is an exoplanet

    Expected input format:
    {
        "mission": "kepler" (optional),
        "period": 3.52,
        "duration": 2.5,
        "depth": 1000,
        "impact": 0.5,
        "snr": 15.2,
        "steff": 5778,
        "srad": 1.0,
        "slogg": 4.5,
        "tmag": 12.5
    }
    """
    try:
        data = None

        # Try to get data from JSON body first
        if request.is_json:
            data = request.json
        elif request.method == 'POST' and request.form:
            # Handle form-encoded data
            data = request.form.to_dict()
        elif request.args and len(request.args) > 0:
            # Handle standard query parameters
            data = request.args.to_dict()
        else:
            # Try to parse malformed query string (client sending JSON-like format)
            data = parse_malformed_query_string(request.query_string.decode('utf-8'))

        # If standard parsing failed but we have a malformed query string, try parsing it
        if (data is None or not data) and request.query_string:
            data = parse_malformed_query_string(request.query_string.decode('utf-8'))

        if data is None or not data:
            return jsonify({
                'error': '400 Bad Request: Failed to decode JSON object: Expecting value: line 1 column 1 (char 0)',
                'message': 'Request body is empty or invalid JSON. Please send data as JSON body, form data, or query parameters.'
            }), 400

        # Convert string values to appropriate types
        for key in data:
            if isinstance(data[key], str):
                try:
                    # Try to convert numeric strings
                    if '.' in data[key]:
                        data[key] = float(data[key])
                    else:
                        data[key] = int(data[key])
                except ValueError:
                    pass  # Keep as string if not numeric

        # Log the received data for debugging
        print(f"📥 Received prediction request: {data}")

        result = predict_exoplanet_internal(data)
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error processing prediction: {e}")
        return jsonify({
            'error': str(e),
            'message': 'Error processing prediction'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Exoplanet Analysis Backend...")
    print(f"📁 Model path: {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=True)
