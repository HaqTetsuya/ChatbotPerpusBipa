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
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("api.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Download required NLTK resources (only needs to run once on server startup)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables for intent classification
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model")
intent_model = None
intent_tokenizer = None
intent_classes = None
intent_thresholds = None

class BookRecommender:
    def __init__(self, model_name='all-minilm-l6-v2'):
        """Initialize the book recommender with specified model."""
        self.model_name = model_name
        self.model = None
        self.book_embeddings = None
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        logger.info(f"BookRecommender initialized with model: {model_name}")
        
    def preprocess_text(self, text):
        """Advanced text preprocessing with stopword removal and lemmatization."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize, remove stopwords, and lemmatize
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)

    def load_model(self, folder_path="recommender_model"):
        """Load a previously saved model and embeddings for inference."""
        try:
            # Check if folder exists
            if not os.path.exists(folder_path):
                logger.error(f"Model folder {folder_path} does not exist.")
                return False
                
            # Load configuration
            config_path = os.path.join(folder_path, "config.pkl")
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            self.model_name = config['model_name']
            logger.info(f"Loaded configuration: model_name={self.model_name}")
            
            # Load the sentence transformer model
            model_path = os.path.join(folder_path, "sentence_transformer")
            self.model = SentenceTransformer(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load book embeddings
            embeddings_path = os.path.join(folder_path, "book_embeddings.pkl")
            with open(embeddings_path, 'rb') as f:
                self.book_embeddings = pickle.load(f)
            logger.info(f"Embeddings loaded: {len(self.book_embeddings)} book vectors")
            
            # Load the DataFrame
            df_path = os.path.join(folder_path, "books_data.pkl")
            with open(df_path, 'rb') as f:
                self.df = pickle.load(f)
            logger.info(f"DataFrame loaded: {len(self.df)} books")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False

    def recommend_books(self, user_query, top_n=5, include_description=True):
        """Recommend books based on user query."""
        if self.model is None or self.book_embeddings is None or self.df is None:
            logger.error("Model not initialized. Cannot make recommendations.")
            return []
            
        logger.info(f"Finding books similar to: '{user_query}'")
        
        try:
            # Preprocess the query the same way as the book text
            processed_query = self.preprocess_text(user_query)
            
            # Encode user query
            user_embedding = self.model.encode([processed_query])
            
            # Compute similarity between query and books
            similarities = cosine_similarity(user_embedding, self.book_embeddings)[0]
            
            # Get top N most similar books
            similar_books_idx = np.argsort(similarities)[-top_n:][::-1]
            
            recommendations = []
            
            for i, idx in enumerate(similar_books_idx):
                book_data = {}
                
                # Extract book information
                if 'Title' in self.df.columns:
                    book_data['title'] = self.df.iloc[idx]['Title']
                
                if 'Authors' in self.df.columns:
                    book_data['author'] = self.df.iloc[idx]['Authors']
                
                if 'Category' in self.df.columns:
                    book_data['category'] = self.df.iloc[idx]['Category']
                
                if 'Publish Date (Year)' in self.df.columns:
                    book_data['year'] = self.df.iloc[idx]['Publish Date (Year)']
                    
                if include_description and 'Description' in self.df.columns:
                    # Truncate long descriptions
                    description = self.df.iloc[idx]['Description']
                    if len(description) > 200:
                        description = description[:197] + "..."
                    book_data['description'] = description
                
                # Add similarity score
                book_data['relevance_score'] = float(similarities[idx])
                book_data['rank'] = i + 1
                
                recommendations.append(book_data)
            
            logger.info(f"Successfully generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
            return []

# Initialize the recommender system
recommender = BookRecommender()
recommender_model_loaded = False

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
        logger.warning(f"Threshold file not found at {threshold_path}. Using default values.")
        return {
            "energy_threshold": 0.0,  # Replace with your default value
            "msp_threshold": 0.5      # Replace with your default value
        }

def load_intent_resources():
    """Load model, tokenizer, intent classes, and thresholds for intent classification."""
    global intent_model, intent_tokenizer, intent_classes, intent_thresholds
    
    logger.info(f"Loading intent resources from {MODEL_SAVE_PATH}...")
    
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

def predict_intent_with_enhanced_ood(text, model, tokenizer, intent_classes, 
                                    energy_threshold, msp_threshold, method='combined'):
    """
    Predict intent with enhanced out-of-distribution detection and print details to terminal.
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
        "energy_score": energy.item()
    }

def initialize_models():
    """Load both models on startup."""
    global recommender_model_loaded
    
    # Load intent classification model
    try:
        load_intent_resources()
        logger.info("Intent classification model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load intent model: {str(e)}", exc_info=True)
    
    # Load book recommender model
    logger.info("Loading recommender model...")
    recommender_model_loaded = recommender.load_model("recommender_model")
    if recommender_model_loaded:
        logger.info("Book recommender model loaded successfully!")
    else:
        logger.error("Failed to load book recommender model.")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint to check if the API is running and models are loaded."""
    global recommender_model_loaded
    
    intent_model_loaded = intent_model is not None and intent_tokenizer is not None
    
    return jsonify({
        "status": "healthy" if (intent_model_loaded and recommender_model_loaded) else "partial",
        "services": {
            "intent_classification": "loaded" if intent_model_loaded else "not loaded",
            "book_recommendations": "loaded" if recommender_model_loaded else "not loaded"
        }
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
    
    # Return prediction as JSON
    return jsonify(result)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Endpoint to get book recommendations based on user query."""
    global recommender_model_loaded

    if not recommender_model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "The recommendation model is not properly loaded."
        }), 503

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Invalid request",
            "message": "No JSON data provided."
        }), 400

    query = data.get('query')
    top_n = data.get('top_n', 5)
    include_description = data.get('include_description', True)

    if not query:
        return jsonify({
            "error": "Missing parameter",
            "message": "Query parameter is required."
        }), 400

    try:
        # Get recommendations
        recommendations = recommender.recommend_books(
            user_query=query,
            top_n=int(top_n),
            include_description=bool(include_description)
        )

        # CLEAN recommendations to make it JSON serializable
        def clean_np(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: clean_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_np(i) for i in obj]
            else:
                return obj

        recommendations_clean = clean_np(recommendations)

        return jsonify({
            "query": query,
            "recommendations": recommendations_clean,
            "count": len(recommendations_clean)
        })

    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Processing error",
            "message": f"An error occurred while processing your request: {str(e)}"
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Endpoint to get statistics about the loaded dataset."""
    global recommender, recommender_model_loaded
    
    if not recommender_model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "The recommendation model is not properly loaded."
        }), 503
    
    try:
        stats = {
            "total_books": len(recommender.df) if recommender.df is not None else 0,
            "model_name": recommender.model_name,
            "categories": list(recommender.df['Category'].value_counts().head(10).to_dict().keys()) 
                         if recommender.df is not None and 'Category' in recommender.df.columns else []
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in stats endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Processing error",
            "message": f"An error occurred while retrieving stats: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint providing API information."""
    return jsonify({
        "status": "API is running",
        "services": {
            "intent_analysis": "Available at /api/analyze",
            "book_recommendations": "Available at /api/recommend",
            "statistics": "Available at /api/stats",
            "health_check": "Available at /api/health"
        },
        "version": "1.0.0"
    })

if __name__ == '__main__':
    # Initialize models when the app starts
    initialize_models()
    
    # Set port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # For development use debug=True, for production use debug=False
    app.run(host='0.0.0.0', port=port, debug=False)