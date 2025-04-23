from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

# Global variables for intent classification
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model")
intent_model = None
intent_tokenizer = None
intent_classes = None
intent_thresholds = None

# Global variables for book recommendation
print("🔄 Loading metadata and embeddings...")
embedding_path = "book_embeddings.npy"
metadata_path = "books_metadata.pkl"
book_embeddings = np.load(embedding_path, mmap_mode='r')  # mmap avoids loading all into RAM
with open(metadata_path, 'rb') as f:
    books_df = pickle.load(f)
# Load lightweight model ONCE for book recommendations
recommendation_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # force CPU
print(f"✅ Loaded {len(books_df)} books. Using CPU mode with memory mapping.")

def load_ood_thresholds(model_path):
    """Load the OOD thresholds from the model directory - using JSON instead of pickle."""
    # Look for JSON file instead of pickle
    threshold_path = os.path.join(model_path, "ood_thresholds.json")
    
    # Check if file exists before attempting to open
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            return json.load(f)
    else:
        # Provide default thresholds if file not found
        print(f"Warning: Threshold file not found at {threshold_path}. Using default values.")
        return {
            "energy_threshold": 0.0,  # Replace with your default value
            "msp_threshold": 0.5      # Replace with your default value
        }

def load_intent_resources():
    """Load model, tokenizer, intent classes, and thresholds for intent classification."""
    global intent_model, intent_tokenizer, intent_classes, intent_thresholds
    
    print(f"Loading intent resources from {MODEL_SAVE_PATH}...")
    
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
    
    print("Intent resources loaded successfully")
    print(f"Loaded {len(intent_classes)} intent classes")
    print(f"Thresholds: {intent_thresholds}")

def predict_intent_with_enhanced_ood(text, model, tokenizer, intent_classes, 
                                    energy_threshold, msp_threshold, method='combined'):
    """
    Predict intent with enhanced out-of-distribution detection and print details to terminal.
    """
    print("\n========== INTENT PREDICTION DEBUG ==========")
    print(f"Input Text: {text}")
    print(f"Detection Method: {method}")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print(f"Logits: {logits.numpy().tolist()}")

    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_prob, pred_idx = torch.max(probs, dim=-1)

    print(f"Softmax Probabilities: {probs.numpy().tolist()}")
    print(f"Max Probability (Confidence): {max_prob.item():.4f}")
    print(f"Predicted Index: {pred_idx.item()}")
    
    # Calculate energy score
    energy = -torch.logsumexp(logits, dim=-1)
    print(f"Energy Score: {energy.item():.4f}")
    
    # OOD detection
    is_ood = False
    if method == 'energy':
        is_ood = energy.item() > energy_threshold
    elif method == 'msp':
        is_ood = max_prob.item() < msp_threshold
    elif method == 'combined':
        is_ood = (energy.item() > energy_threshold) and (max_prob.item() < msp_threshold)
    
    print(f"OOD Detection -> is_ood: {is_ood}")
    if is_ood:
        print("Prediction marked as OUT-OF-DISTRIBUTION.")
    else:
        print("Prediction marked as IN-DISTRIBUTION.")
    
    # Get intent label
    predicted_intent = intent_classes[pred_idx.item()] if not is_ood else "unknown"
    print(f"Predicted Intent: {predicted_intent}")
    print("=============================================\n")

    return {
        "intent": predicted_intent,
        "is_ood": is_ood,
        "confidence": max_prob.item(),
        "energy_score": energy.item()
    }


def get_book_recommendations(user_query, top_n=5):
    """Get book recommendations based on user query."""
    user_embedding = recommendation_model.encode([user_query], convert_to_numpy=True, device='cpu')
    similarities = cosine_similarity(user_embedding, book_embeddings)[0]
    similar_books_idx = np.argsort(similarities)[-top_n:][::-1]
    recommendations = []
    for idx in similar_books_idx:
        book = {
            "title": str(books_df.iloc[idx].get('Title', 'Unknown')),
            "author": str(books_df.iloc[idx].get('Authors', 'Unknown')),
            "category": str(books_df.iloc[idx].get('Category', 'N/A')),
            "description": str(books_df.iloc[idx].get('Description', 'N/A')),
            "year": int(books_df.iloc[idx].get('Publish Date (Year)', 0)) if not pd.isna(books_df.iloc[idx].get('Publish Date (Year)')) else None,
            "relevance": round(float(similarities[idx]), 2)
        }
        recommendations.append(book)
    return recommendations

@app.route('/analyze', methods=['POST'])
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
    
    # Return prediction as JSON
    return jsonify(result)

@app.route("/recommend", methods=["POST"])
def recommend():
    """Endpoint to get book recommendations."""
    data = request.get_json()
    user_query = data.get("query")
    top_n = int(data.get("top_n", 5))
    if not user_query:
        return jsonify({"error": "Query is required."}), 400
    try:
        results = get_book_recommendations(user_query, top_n)
        return jsonify({"query": user_query, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Simple test endpoint for intent analysis
@app.route('/test-intent', methods=['GET'])
def test_intent():
    return jsonify({
        "status": "Intent analysis API is running", 
        "model_path": MODEL_SAVE_PATH
    })

# Health check for book recommendations
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "API is running",
        "services": {
            "intent_analysis": "Available at /analyze",
            "book_recommendations": "Available at /recommend"
        }
    })

if __name__ == '__main__':
    # Load intent resources only once before starting the server
    load_intent_resources()
    
    # Start the server
    # Option 1: Keep debug mode but disable auto-reloader
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    
    # Option 2 (Alternative): Disable debug mode entirely for production
    # app.run(debug=False, host='0.0.0.0', port=5000)
    # app.run(debug=False, host='0.0.0.0', port=5000)
#curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '{"text": "buku fantasi romance"}'	