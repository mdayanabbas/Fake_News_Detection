import pandas as pd
import numpy as np
import joblib
import re
import os
import sys
import traceback
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from lime import lime_text
from lime.lime_text import LimeTextExplainer

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('app_debug.log'),
                        logging.StreamHandler(sys.stdout)
                    ])

# Set the NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Download necessary NLTK resources
logging.info("Downloading NLTK resources...")
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {str(e)}")

app = Flask(__name__)

# Robust model loading with error handling
def safe_load_model(filepath):
    try:
        logging.info(f"Attempting to load model from {filepath}")
        model = joblib.load(filepath)
        logging.info(f"Successfully loaded model from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {filepath}: {str(e)}")
        logging.error(traceback.format_exc())
        return None

# Initialize DistilBERT flag and model variables
distilbert_available = False
distilbert_model = None
distilbert_tokenizer = None

# Try to initialize DistilBERT model and tokenizer
def try_initialize_distilbert():
    global distilbert_available, distilbert_model, distilbert_tokenizer
    try:
        logging.info("Attempting to initialize DistilBERT model and tokenizer")
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        import torch
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        
        distilbert_model = model
        distilbert_tokenizer = tokenizer
        distilbert_available = True
        logging.info("DistilBERT model and tokenizer initialized successfully")
    except Exception as e:
        distilbert_available = False
        logging.error(f"Error initializing DistilBERT: {str(e)}")
        logging.error(traceback.format_exc())
        logging.warning("DistilBERT will not be available for predictions")


try_initialize_distilbert()

def predict_with_distilbert(text):
    if not distilbert_available:
        return "DistilBERT not available", 0.5, 0.5
    
    try:
        logging.info("Predicting with DistilBERT")
        import torch
        # Clean text
        cleaned_text = clean_text(text)
        
        # Tokenize
        inputs = distilbert_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Make prediction
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        # Convert probabilities to numpy for easier handling
        probs = probabilities.numpy()[0]
        fake_prob, real_prob = probs[0], probs[1]
        
        prediction_label = 'FAKE' if predicted_class == 0 else 'REAL'
        logging.info(f"DistilBERT prediction: {prediction_label}")
        
        return prediction_label, fake_prob, real_prob
    except Exception as e:
        logging.error(f"DistilBERT prediction error: {str(e)}")
        logging.error(traceback.format_exc())
        return "Error", 0.5, 0.5

# Function to make DistilBERT compatible with LIME (if available)
def distilbert_predict_proba(texts):
    if not distilbert_available:
        return np.array([[0.5, 0.5]] * len(texts))
    
    try:
        import torch
        results = []
        for text in texts:
            try:
                cleaned_text = clean_text(text)
                inputs = distilbert_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                
                with torch.no_grad():
                    outputs = distilbert_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                
                # Convert to numpy array
                probs = probabilities.numpy()[0]
                results.append(probs)
            except Exception as e:
                logging.error(f"Error in distilbert_predict_proba: {str(e)}")
                results.append(np.array([0.5, 0.5]))  # Default to 50-50 on error
        
        return np.array(results)
    except Exception as e:
        logging.error(f"Global error in distilbert_predict_proba: {str(e)}")
        return np.array([[0.5, 0.5]] * len(texts))

# Load pre-trained pipeline models with detailed logging
model_files = {
    "Logistic Regression": 'logi.joblib',
    "Random Forest": 'random_forest.joblib',
    "Naive Bayes": 'naive.joblib',
    "SVM": 'nb_model.joblib',
    "Dense Model": 'dense.pkl'
}

vectorizer_files = {
    "Dense Model": 'vectorize_model.pkl'  # Vectorizer for the Dense model
}

model_pipelines = {}
vectorizers = {}

# Load both model pipelines and vectorizers
for model_name, filepath in model_files.items():
    pipeline = safe_load_model(filepath)
    if pipeline is not None:
        model_pipelines[model_name] = pipeline
    else:
        logging.warning(f"Could not load {model_name} pipeline")

for model_name, filepath in vectorizer_files.items():
    vectorizer = safe_load_model(filepath)
    if vectorizer is not None:
        vectorizers[model_name] = vectorizer
    else:
        logging.warning(f"Could not load vectorizer for {model_name}")

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocessing text function
def preprocess_text(text):
    try:
        text = clean_text(text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Preprocessing error: {str(e)}")
        logging.error(traceback.format_exc())
        return text

# Function to get LIME explanation
def explain_with_lime(text, model_choice):
    logging.info(f"Explaining with LIME for {model_choice}")
    
    # Initialize LimeTextExplainer
    explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
    
    try:
        # Special handling for DistilBERT
        if model_choice == "DistilBERT":
            if not distilbert_available:
                return "DistilBERT model not available for explanation"
            
            explanation = explainer.explain_instance(
                clean_text(text),
                distilbert_predict_proba,
                num_features=10
            )
        else:
            # Select the appropriate model pipeline
            pipeline = model_pipelines.get(
                model_choice,
                next(iter(model_pipelines.values()))  # Default to first available model if not found
            )
            
            # Preprocess the text for explanation
            processed_text = preprocess_text(text)
            
            # Use vectorizer if Dense Model is chosen
            if model_choice == "Dense Model" and model_choice in vectorizers:
                vectorizer = vectorizers[model_choice]
                processed_text = vectorizer.transform([processed_text]).toarray()
            else:
                processed_text = [processed_text]
            
            # Explain the prediction with LIME
            explanation = explainer.explain_instance(
                processed_text[0],  # Provide single text instance for explanation
                pipeline.predict_proba,  # Model's prediction function
                num_features=10  # Number of features to show in explanation
            )
        
        # Convert explanation to a readable format
        explanation_html = explanation.as_html()
        logging.info("Generated LIME explanation.")
        
        # Add debug to check if explanation_html is being generated properly
        logging.debug(f"Explanation HTML: {explanation_html[:500]}...")  # Preview the start of the HTML

        return explanation_html

    except Exception as e:
        logging.error(f"LIME explanation error for {model_choice}: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error in LIME explanation: {str(e)}"

# Home route for the application
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', models=list(model_pipelines.keys()) + (["DistilBERT"] if distilbert_available else []))

# Prediction route
def predict(text, model_choice):
    logging.info(f"Predicting with {model_choice}")
    
    # Special handling for DistilBERT
    if model_choice == "DistilBERT":
        if not distilbert_available:
            logging.error("DistilBERT model not available")
            return {"prediction": "DistilBERT model not available", "probabilities": None}
        
        prediction_label, fake_prob, real_prob = predict_with_distilbert(text)
        return {
            "prediction": prediction_label,
            "probabilities": {
                "FAKE": round(fake_prob * 100, 2),
                "REAL": round(real_prob * 100, 2)
            }
        }
    
    # Check if models are loaded
    if not model_pipelines:
        logging.error("No models have been successfully loaded")
        return {"prediction": "No models available", "probabilities": None}

    # Default to first available model if chosen model not found
    pipeline = model_pipelines.get(
        model_choice, 
        next(iter(model_pipelines.values()))
    )

    try:
        processed_text = preprocess_text(text)
        logging.info(f"Processed text: {processed_text}")

        # Use vectorizer if Dense model is chosen
        if model_choice == "Dense Model" and model_choice in vectorizers:
            vectorizer = vectorizers[model_choice]
            processed_text = vectorizer.transform([processed_text]).toarray()
        else:
            processed_text = [processed_text]

        # Predict label
        prediction = pipeline.predict(processed_text)[0]
        
        # Get probabilities
        try:
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba(processed_text)[0]
                fake_prob, real_prob = probabilities[0], probabilities[1]
                prediction_label = 'REAL' if real_prob > fake_prob else 'FAKE'
            else:
                fake_prob = real_prob = 0.5
                prediction_label = 'REAL'
        except Exception as e:
            logging.error(f"Probability extraction error: {str(e)}")
            fake_prob = real_prob = 0.5
            prediction_label = 'REAL'

        logging.info(f"Prediction: {prediction_label}")
        return {
            "prediction": prediction_label, 
            "probabilities": {
                "FAKE": round(fake_prob * 100, 2),
                "REAL": round(real_prob * 100, 2)
            }
        }
    
    except Exception as e:
        logging.error(f"Prediction error for {model_choice}: {str(e)}")
        logging.error(traceback.format_exc())
        return {"prediction": f"Error in prediction: {str(e)}", "probabilities": None}

@app.route('/', methods=['POST'])
def webapp():
    try:
        text = request.form['text']
        model_choice = request.form['model_choice']
        result = predict(text, model_choice)
        explanation = explain_with_lime(text, model_choice)  # Get LIME explanation
        return render_template('index.html', 
                               text=text, 
                               result=result['prediction'], 
                               model_choice=model_choice,
                               probabilities=result.get('probabilities'),
                               explanation=explanation,
                               models=list(model_pipelines.keys()) + (["DistilBERT"] if distilbert_available else []))
    except Exception as e:
        logging.error(f"Web app error: {str(e)}")
        logging.error(traceback.format_exc())
        return "An error occurred", 500

# API route for prediction
@app.route('/predict/', methods=['GET', 'POST'])
def api():
    try:
        text = request.args.get("text")
        model_choice = request.args.get("model_choice")
        prediction = predict(text, model_choice)
        return jsonify(prediction)
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify(error="An error occurred"), 500

# Function to save DistilBERT model locally
@app.route('/save-distilbert', methods=['GET'])
def save_distilbert_model():
    try:
        if distilbert_available:
            # Create directory if it doesn't exist
            os.makedirs('distilbert_model', exist_ok=True)
            
            # Save model and tokenizer
            distilbert_model.save_pretrained('distilbert_model')
            distilbert_tokenizer.save_pretrained('distilbert_model')
            
            return jsonify({"status": "success", "message": "DistilBERT model saved successfully"})
        else:
            return jsonify({"status": "error", "message": "DistilBERT model not available"})
    except Exception as e:
        logging.error(f"Error saving DistilBERT model: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    # Log available models
    logging.info("Available Models:")
    for name in model_pipelines.keys():
        logging.info(f"- {name}")
    
    if distilbert_available:
        logging.info("- DistilBERT")
    
    app.run(host="0.0.0.0", port=8000, debug=True)