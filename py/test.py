from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pickle
import os
import json
import logging
import csv
from datetime import datetime
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("intent_api.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables for intent classification
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model")
intent_model = None
intent_tokenizer = None
intent_classes = None
intent_thresholds = None

# CSV file for evaluation data
EVAL_CSV = "model_evaluation.csv"

def setup_evaluation_csv():
    """Set up the CSV file for tracking model performance"""
    if not os.path.exists(EVAL_CSV):
        with open(EVAL_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 
                'input_text', 
                'predicted_intent', 
                'is_ood', 
                'confidence', 
                'energy_score',
                'detection_method'
            ])
        logger.info(f"Created evaluation CSV file: {EVAL_CSV}")

def save_prediction_to_csv(input_text, result, method):
    """Save prediction results to CSV for later analysis"""
    with open(EVAL_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_text,
            result['intent'],
            result['is_ood'],
            result['confidence'],
            result['energy_score'],
            method
        ])

def load_ood_thresholds(model_path):
    """Load the OOD thresholds from the model directory"""
    threshold_path = os.path.join(model_path, "ood_thresholds.json")
    
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            return json.load(f)
    else:
        # Provide default thresholds if file not found
        logger.warning(f"Threshold file not found at {threshold_path}. Using default values.")
        return {
            "energy_threshold": 0.0,  # Replace with your default value
            "msp_threshold": 0.5      # Replace with your default value
        }

def load_intent_resources():
    """Load model, tokenizer, intent classes, and thresholds for intent classification."""
    global intent_model, intent_tokenizer, intent_classes, intent_thresholds
    
    logger.info(f"Loading intent resources from {MODEL_SAVE_PATH}...")
    
    try:
        # Load model and tokenizer
        intent_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
        intent_tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
        
        # Load intent classes
        intent_classes_path = os.path.join(MODEL_SAVE_PATH, "intent_classes.pkl")
        if os.path.exists(intent_classes_path):
            with open(intent_classes_path, "rb") as f:
                intent_classes = pickle.load(f)
        else:
            raise FileNotFoundError(f"Intent classes file not found at {intent_classes_path}")
        
        # Load OOD thresholds
        intent_thresholds = load_ood_thresholds(MODEL_SAVE_PATH)
        
        logger.info("Intent resources loaded successfully")
        logger.info(f"Loaded {len(intent_classes)} intent classes")
        logger.info(f"Thresholds: {intent_thresholds}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load intent resources: {str(e)}", exc_info=True)
        return False

def predict_intent_with_enhanced_ood(text, model, tokenizer, intent_classes, 
                                    energy_threshold, msp_threshold, method='combined'):
    """
    Predict intent with enhanced out-of-distribution detection and detailed logging.
    """
    logger.info("\n========== INTENT PREDICTION DEBUG ==========")
    logger.info(f"Input Text: {text}")
    logger.info(f"Detection Method: {method}")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    logger.info(f"Logits: {logits.numpy().tolist()}")

    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_prob, pred_idx = torch.max(probs, dim=-1)

    logger.info(f"Softmax Probabilities: {probs.numpy().tolist()}")
    logger.info(f"Max Probability (Confidence): {max_prob.item():.4f}")
    logger.info(f"Predicted Index: {pred_idx.item()}")
    
    # Calculate energy score
    energy = -torch.logsumexp(logits, dim=-1)
    logger.info(f"Energy Score: {energy.item():.4f}")
    
    # OOD detection
    is_ood = False
    if method == 'energy':
        is_ood = energy.item() > energy_threshold
    elif method == 'msp':
        is_ood = max_prob.item() < msp_threshold
    elif method == 'combined':
        is_ood = (energy.item() > energy_threshold) and (max_prob.item() < msp_threshold)
    
    logger.info(f"OOD Detection -> is_ood: {is_ood}")
    if is_ood:
        logger.info("Prediction marked as OUT-OF-DISTRIBUTION.")
    else:
        logger.info("Prediction marked as IN-DISTRIBUTION.")
    
    # Get intent label
    predicted_intent = intent_classes[pred_idx.item()] if not is_ood else "unknown"
    logger.info(f"Predicted Intent: {predicted_intent}")
    logger.info("=============================================\n")

    return {
        "intent": predicted_intent,
        "is_ood": is_ood,
        "confidence": max_prob.item(),
        "energy_score": energy.item(),
        # Add all class probabilities for detailed analysis
        "class_probabilities": {
            intent_classes[i]: float(prob) 
            for i, prob in enumerate(probs[0].numpy())
        }
    }

def initialize_models():
    """Load intent classification model on startup."""
    # Create evaluation CSV if it doesn't exist
    setup_evaluation_csv()
    
    # Load intent classification model
    model_loaded = load_intent_resources()
    if model_loaded:
        logger.info("Intent classification model loaded successfully!")
        return True
    else:
        logger.error("Failed to load intent model.")
        return False

@app.route('/', methods=['GET'])
def index():
    """Simple HTML form for testing the API interactively"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint to check if the API is running and models are loaded."""
    models_loaded = intent_model is not None and intent_tokenizer is not None
    
    return jsonify({
        "status": "healthy" if models_loaded else "unhealthy",
        "intent_model_loaded": models_loaded,
        "available_endpoints": ["/", "/api/health", "/api/analyze", "/api/stats"]
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Endpoint to predict intent from text."""
    # Check if request contains JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get text from request
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data['text']
    
    # Default to combined method unless specified
    method = data.get('method', 'combined')
    if method not in ['energy', 'msp', 'combined']:
        return jsonify({"error": "Invalid method. Must be 'energy', 'msp', or 'combined'"}), 400
    
    # Make prediction
    result = predict_intent_with_enhanced_ood(
        text, 
        intent_model, 
        intent_tokenizer, 
        intent_classes, 
        intent_thresholds["energy_threshold"],
        intent_thresholds["msp_threshold"],
        method=method
    )
    
    # Save result to CSV for evaluation
    save_prediction_to_csv(text, result, method)
    
    # Return prediction as JSON
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about model usage and predictions."""
    try:
        stats = {
            "model_info": {
                "num_intent_classes": len(intent_classes) if intent_classes else 0,
                "model_path": MODEL_SAVE_PATH,
                "thresholds": intent_thresholds
            },
            "usage": {}
        }
        
        # Read CSV to generate statistics if it exists
        if os.path.exists(EVAL_CSV):
            with open(EVAL_CSV, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                stats["usage"] = {
                    "total_queries": len(rows),
                    "ood_count": sum(1 for row in rows if row["is_ood"] == "True"),
                    "top_intents": {}
                }
                
                # Count intents for statistical analysis
                intent_counts = {}
                for row in rows:
                    intent = row["predicted_intent"]
                    if intent not in intent_counts:
                        intent_counts[intent] = 0
                    intent_counts[intent] += 1
                
                # Get top 5 intents
                top_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                stats["usage"]["top_intents"] = dict(top_intents)
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in stats endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Processing error",
            "message": f"An error occurred while retrieving stats: {str(e)}"
        }), 500

# Add this new endpoint to download the evaluation data
@app.route('/api/download_eval_data', methods=['GET'])
def download_eval_data():
    """Return the evaluation data as JSON for analysis"""
    try:
        if not os.path.exists(EVAL_CSV):
            return jsonify({"error": "No evaluation data available yet"}), 404
            
        with open(EVAL_CSV, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        return jsonify({
            "count": len(rows),
            "data": rows
        })
        
    except Exception as e:
        logger.error(f"Error downloading evaluation data: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Processing error",
            "message": f"An error occurred: {str(e)}"
        }), 500



if __name__ == '__main__':
    # Create templates directory and HTML template if they don't exist
    templates_dir = os.path.join(BASE_DIR, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create a simple HTML template for testing
    with open(os.path.join(templates_dir, "index.html"), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Intent Classification Tester</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        textarea { width: 100%; height: 100px; }
        select { padding: 5px; }
        button { padding: 8px 15px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        #result { margin-top: 20px; border: 1px solid #ddd; padding: 15px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Intent Classification Tester</h1>
    <div class="form-group">
        <label for="text">Enter text to classify:</label>
        <textarea id="text" placeholder="Type your text here..."></textarea>
    </div>
    <div class="form-group">
        <label for="method">OOD Detection Method:</label>
        <select id="method">
            <option value="combined">Combined (Energy + MSP)</option>
            <option value="energy">Energy Based</option>
            <option value="msp">Maximum Softmax Probability</option>
        </select>
    </div>
    <button onclick="analyzeText()">Analyze Intent</button>
    
    <h2>Result:</h2>
    <pre id="result">Results will appear here...</pre>
    
    <script>
        async function analyzeText() {
            const text = document.getElementById('text').value;
            const method = document.getElementById('method').value;
            
            if (!text) {
                alert('Please enter some text to analyze');
                return;
            }
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text, method }),
                });
                
                const data = await response.json();
                document.getElementById('result').textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                document.getElementById('result').textContent = 'Error: ' + error.message;
            }
        }
    </script>
</body>
</html>
        """)

    # Initialize models when the app starts
    model_loaded = initialize_models()
    
    # Set port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # For development use debug=True, for production use debug=False
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
