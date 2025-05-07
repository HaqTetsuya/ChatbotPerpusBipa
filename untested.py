import torch
import os
import json
import pickle
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import IPython.display as display
from IPython.display import HTML, display

# Mount Google Drive
drive.mount('/content/drive')

class IntentPredictor:
    """Class for intent prediction with OOD detection for Google Colab"""
    
    def __init__(self, model_path, mount_drive=True):
        """Initialize the predictor with a trained model path"""
        if mount_drive and not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load intent classes and label encoder
        with open(os.path.join(model_path, "intent_classes.pkl"), "rb") as f:
            self.intent_classes = pickle.load(f)
        
        with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
            
        # Load OOD thresholds
        self.thresholds = self.load_ood_thresholds()
        
        print(f"Model loaded successfully. Detected {len(self.intent_classes)} intents.")
        print(f"Supported intents: {', '.join(self.intent_classes)}")
        print(f"OOD thresholds: Energy={self.thresholds['energy_threshold']:.4f}, MSP={self.thresholds['msp_threshold']:.4f}")
    
    def load_ood_thresholds(self):
        """Load OOD detection thresholds from file"""
        try:
            with open(os.path.join(self.model_path, "ood_thresholds.json"), 'r') as f:
                thresholds = json.load(f)
                return thresholds
        except FileNotFoundError:
            try:
                with open(os.path.join(self.model_path, "ood_threshold.json"), 'r') as f:
                    threshold_data = json.load(f)
                    return {
                        "energy_threshold": threshold_data["energy_threshold"],
                        "msp_threshold": 0.5  # Default value
                    }
            except FileNotFoundError:
                print("Warning: OOD threshold files not found. Using default thresholds.")
                return {
                    "energy_threshold": 0.0,
                    "msp_threshold": 0.5
                }
    
    def predict(self, text, temperature=1.0):
        """Predict intent for a single text with OOD detection"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        
        # Calculate energy score for OOD detection
        energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        energy_score = energy.item()
        
        # Calculate maximum softmax probability
        softmax_probs = F.softmax(logits, dim=1)
        max_prob, prediction = torch.max(softmax_probs, dim=1)
        max_prob = max_prob.item()
        prediction = prediction.item()
        
        # Get predicted intent
        intent = self.intent_classes[prediction]
        
        # Get all probabilities with intent names
        all_probs = softmax_probs[0].cpu().numpy()
        intent_probs = {self.intent_classes[i]: float(all_probs[i]) for i in range(len(self.intent_classes))}
        
        # Determine if sample is OOD using both methods
        is_ood_energy = energy_score > self.thresholds["energy_threshold"]
        is_ood_msp = max_prob < self.thresholds["msp_threshold"]
        
        # Use either one or both methods for final OOD decision
        # Here we're using energy method as primary, MSP as secondary validation
        is_ood = is_ood_energy or is_ood_msp
        
        # Return full prediction results
        result = {
            "text": text,
            "intent": intent,
            "is_ood": is_ood,
            "confidence": max_prob,
            "energy_score": energy_score,
            "intent_probabilities": intent_probs,
            "ood_details": {
                "is_ood_energy": is_ood_energy,
                "is_ood_msp": is_ood_msp,
                "energy_threshold": self.thresholds["energy_threshold"],
                "msp_threshold": self.thresholds["msp_threshold"]
            }
        }
        
        return result
    
    def predict_batch(self, texts):
        """Predict intents for a batch of texts"""
        results = []
        for text in tqdm(texts, desc="Predicting intents"):
            results.append(self.predict(text))
        return results
    
    def predict_csv(self, csv_path, text_column="text", save_path=None):
        """Predict intents for texts in a CSV file and optionally save results"""
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        # Make predictions
        texts = df[text_column].tolist()
        results = self.predict_batch(texts)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Merge with original data if needed
        if len(df.columns) > 1:
            df['predicted_intent'] = results_df['intent']
            df['confidence'] = results_df['confidence']
            df['is_ood'] = results_df['is_ood']
            output_df = df
        else:
            output_df = results_df
        
        # Save results if path is provided
        if save_path:
            output_df.to_csv(save_path, index=False)
            print(f"Prediction results saved to {save_path}")
        
        return output_df
    
    def display_prediction_results(self, result, show_top_n=3):
        """Display prediction results in a formatted way in Google Colab"""
        display(HTML(f"""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #333;">Prediction Results</h3>
            <p><b>Text:</b> {result['text']}</p>
            <p><b>Predicted Intent:</b> <span style="color: #4285F4; font-weight: bold;">{result['intent']}</span></p>
            <p><b>Confidence:</b> {result['confidence']:.4f}</p>
            
            <div style="margin-top: 10px;">
                <b>Status:</b> 
                <span style="color: {'#DB4437' if result['is_ood'] else '#0F9D58'}; font-weight: bold;">
                    {'OUT-OF-DISTRIBUTION' if result['is_ood'] else 'In-distribution'}
                </span>
            </div>
            
            <div style="margin-top: 15px;">
                <b>OOD Detection Details:</b>
                <ul>
                    <li>Energy Score: {result['energy_score']:.4f} (Threshold: {result['ood_details']['energy_threshold']:.4f})</li>
                    <li>MSP Score: {result['confidence']:.4f} (Threshold: {result['ood_details']['msp_threshold']:.4f})</li>
                </ul>
            </div>
        </div>
        """))
        
        # Plot top intent probabilities
        top_intents = sorted(result['intent_probabilities'].items(), key=lambda x: x[1], reverse=True)[:show_top_n]
        labels = [intent for intent, _ in top_intents]
        values = [prob for _, prob in top_intents]
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, values, color='skyblue')
        plt.title('Top Intent Probabilities')
        plt.xlabel('Intent')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, true_intents, predicted_intents):
        """Plot confusion matrix for batch predictions"""
        from sklearn.metrics import confusion_matrix
        
        # Get unique intents
        unique_intents = list(set(true_intents + predicted_intents))
        
        # Create confusion matrix
        cm = confusion_matrix(true_intents, predicted_intents, labels=unique_intents)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_intents, yticklabels=unique_intents)
        plt.title('Confusion Matrix')
        plt.ylabel('True Intent')
        plt.xlabel('Predicted Intent')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Function to demonstrate usage with sample texts
def demo_predictor(model_path, sample_texts=None):
    """Demonstrate the intent predictor with sample texts"""
    predictor = IntentPredictor(model_path)
    
    # Use default samples if none provided
    if sample_texts is None:
        sample_texts = [
            "Tolong jelaskan cara membeli tiket",
            "Bagaimana cara mendaftar akun?",
            "Saya ingin membatalkan pesanan saya",
            "kjhgfdtyuiolkjhgfdsdfghjkl",  # Gibberish text to test OOD
            "What is the weather like today?"  # English text to test OOD
        ]
    
    print("\nPredicting sample texts:")
    for text in sample_texts:
        result = predictor.predict(text)
        predictor.display_prediction_results(result)

# Google Colab-specific interactive demo with widgets
def interactive_demo(model_path):
    """Interactive demo using Colab widgets"""
    from ipywidgets import widgets
    from IPython.display import display, clear_output
    
    predictor = IntentPredictor(model_path)
    
    # Create text input widget
    text_input = widgets.Text(
        value='',
        placeholder='Type your text here',
        description='Input:',
        disabled=False,
        layout=widgets.Layout(width='70%')
    )
    
    # Create predict button
    predict_button = widgets.Button(
        description='Predict',
        disabled=False,
        button_style='primary',
        tooltip='Click to predict intent',
        icon='check'
    )
    
    output = widgets.Output()
    
    # Define button click handler
    def on_button_clicked(b):
        with output:
            clear_output()
            if text_input.value.strip():
                result = predictor.predict(text_input.value)
                predictor.display_prediction_results(result)
            else:
                print("Please enter some text to predict.")
    
    predict_button.on_click(on_button_clicked)
    
    # Display widgets
    display(HTML('<h2>Intent Prediction Demo</h2>'))
    display(widgets.HBox([text_input, predict_button]))
    display(output)

# Run prediction from a CSV file including visualization
def predict_from_csv(model_path, csv_path, text_column="text", label_column=None, output_path=None):
    """Run predictions on a CSV file with visualization"""
    predictor = IntentPredictor(model_path)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    texts = df[text_column].tolist()
    
    # Make predictions
    results = predictor.predict_batch(texts)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Merge with original data
    df['predicted_intent'] = results_df['intent']
    df['confidence'] = results_df['confidence']
    df['is_ood'] = results_df['is_ood']
    
    # Save if output path specified
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    # Display summary
    print(f"Total predictions: {len(texts)}")
    print(f"In-distribution: {(~df['is_ood']).sum()} ({(~df['is_ood']).mean()*100:.2f}%)")
    print(f"Out-of-distribution: {df['is_ood'].sum()} ({df['is_ood'].mean()*100:.2f}%)")
    
    # Visualize intent distribution
    plt.figure(figsize=(12, 6))
    intent_counts = df['predicted_intent'].value_counts()
    sns.barplot(x=intent_counts.index, y=intent_counts.values)
    plt.title('Predicted Intent Distribution')
    plt.xlabel('Intent')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Plot confusion matrix if ground truth available
    if label_column and label_column in df.columns:
        predictor.plot_confusion_matrix(
            df[label_column].tolist(),
            df['predicted_intent'].tolist()
        )
    
    return df

# Main execution for Google Colab
# This function is designed to be the main entry point for the colab notebook
def run_predictor_colab():
    print("Intent Prediction with OOD Detection")
    print("====================================")
    
    # Path to the model - adjust as needed
    model_base_path = "/content/drive/MyDrive/models"
    
    # Check if models directory exists
    if not os.path.exists(model_base_path):
        print(f"Warning: Models directory {model_base_path} not found.")
        # Ask user to provide model path
        model_base_path = input("Enter path to models directory: ")
    
    # Get list of available models
    available_models = [d for d in os.listdir(model_base_path) 
                      if os.path.isdir(os.path.join(model_base_path, d))]
    
    if not available_models:
        print("No model directories found. Please provide a path to a model directory.")
        model_path = input("Enter path to model: ")
    else:
        print("\nAvailable models:")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        
        model_idx = int(input("\nSelect a model by number: ")) - 1
        model_path = os.path.join(model_base_path, available_models[model_idx])
    
    print(f"\nUsing model at {model_path}")
    
    # Create predictor
    predictor = IntentPredictor(model_path)
    
    # Choose mode
    print("\nSelect operation mode:")
    print("1. Interactive Demo (with widgets)")
    print("2. Predict from CSV")
    print("3. Demo with sample texts")
    print("4. Predict a single text")
    
    mode = int(input("\nEnter mode number: "))
    
    if mode == 1:
        interactive_demo(model_path)
    elif mode == 2:
        csv_path = input("Enter path to CSV file: ")
        text_column = input("Enter text column name (default: 'text'): ") or "text"
        label_column = input("Enter ground truth column name (optional, for confusion matrix): ")
        if not label_column.strip():
            label_column = None
        output_path = input("Enter output CSV path (optional): ")
        if not output_path.strip():
            output_path = None
        predict_from_csv(model_path, csv_path, text_column, label_column, output_path)
    elif mode == 3:
        demo_predictor(model_path)
    elif mode == 4:
        while True:
            text = input("\nEnter text to predict (or 'quit' to exit): ")
            if text.lower() == 'quit':
                break
            result = predictor.predict(text)
            predictor.display_prediction_results(result)
    else:
        print("Invalid mode selected.")

# Code to run directly in Google Colab
if __name__ == "__main__":
    run_predictor_colab()
	
	
	
# Copy and paste the entire code from above

# Run the interactive demo
run_predictor_colab()

# Alternatively, you can directly use the predictor:
MODEL_PATH = "/content/drive/MyDrive/models/your_model_name"
predictor = IntentPredictor(MODEL_PATH)

# Predict a single text
result = predictor.predict("Bagaimana cara mendaftar akun?")
predictor.display_prediction_results(result)

# Or predict from a CSV file
df = predict_from_csv(MODEL_PATH, "/content/drive/MyDrive/data.csv", "text_column")	