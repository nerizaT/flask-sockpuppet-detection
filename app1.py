import os
import webbrowser
import torch
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from threading import Timer

# Download necessary NLTK resources (first run only)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Flask app setup
app = Flask(__name__, template_folder="html")

# Paths
MODEL_PATH = "model/random_forest.pkl"
FEATURE_METADATA_PATH = "model/feature_metadata.pkl"
EMBEDDINGS_LOOKUP_PATH = "model/embeddings_lookup.csv"
PCA_COMPONENTS_PATH = "model/pca_components.npy"

# Load trained model
print("[INFO] Loading model...")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
else:
    raise FileNotFoundError(f"[ERROR] Model file not found: {MODEL_PATH}")

# Load feature metadata
print("[INFO] Loading feature metadata...")
if os.path.exists(FEATURE_METADATA_PATH):
    feature_metadata = joblib.load(FEATURE_METADATA_PATH)
    feature_names = feature_metadata["features"]
    print("[INFO] Feature metadata loaded.")
else:
    raise FileNotFoundError(f"[ERROR] Feature metadata file not found: {FEATURE_METADATA_PATH}")

# Load embeddings lookup table
if os.path.exists(EMBEDDINGS_LOOKUP_PATH):
    print("[INFO] Loading embeddings lookup table...")
    embeddings_lookup = pd.read_csv(EMBEDDINGS_LOOKUP_PATH)
    embeddings_lookup.set_index("edit_text", inplace=True)
    print("[INFO] Embeddings lookup table loaded.")
else:
    print("[WARNING] Embeddings lookup table not found. Dynamic embedding generation will be used.")
    embeddings_lookup = None

# Load Sentence-BERT model for dynamic embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Load PCA model
if os.path.exists(PCA_COMPONENTS_PATH):
    print("[INFO] Loading PCA components...")
    pca = PCA(n_components=100)
    pca.components_ = np.load(PCA_COMPONENTS_PATH)
    print("[INFO] PCA components loaded.")
else:
    print("[WARNING] PCA components file not found. PCA will be trained dynamically.")
    pca = PCA(n_components=100)

# Preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Clean Wikipedia-style text by removing links, special characters, and extra spaces."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"--\s*<signature>", "", text)
    text = re.sub(r"#REDIRECT \[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def remove_stopwords(text):
    """Remove stopwords from the text."""
    tokens = word_tokenize(text)
    return " ".join(word for word in tokens if word not in stop_words)

def lemmatize_text(text):
    """Lemmatize words in the text."""
    tokens = word_tokenize(text)
    return " ".join(lemmatizer.lemmatize(word) for word in tokens)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json if request.is_json else request.form
    edit_text = data.get("input_text", "").strip()

    if not edit_text:
        return jsonify({"error": "Missing 'input_text' in request."}), 400

    # Apply text preprocessing
    cleaned_text = clean_text(edit_text)
    cleaned_text = remove_stopwords(cleaned_text)
    cleaned_text = lemmatize_text(cleaned_text)

    if not cleaned_text:
        return jsonify({"error": "Processed text is empty after cleaning. Please provide more meaningful input."}), 400

    # Retrieve embeddings from lookup table
    if embeddings_lookup is not None and cleaned_text in embeddings_lookup.index:
        print("[INFO] Using precomputed embeddings.")
        features = embeddings_lookup.loc[cleaned_text].values.reshape(1, -1)
    else:
        print(f"[INFO] Generating new embeddings for: {cleaned_text}")
        embeddings = sbert_model.encode([cleaned_text], convert_to_numpy=True, device=device)

        if pca.components_ is None:
            print("[INFO] Fitting PCA dynamically...")
            pca.fit(embeddings)  # Fit PCA on the fly if needed

        features = pca.transform(embeddings)  # Reduce to 100D
        features = features / np.linalg.norm(features)  # Normalize the embeddings

    print("[DEBUG] Generated Embeddings Shape:", features.shape)
    print("[DEBUG] Generated Embeddings:", features)

    # Compute sentiment features
    polarity = TextBlob(cleaned_text).sentiment.polarity
    subjectivity = TextBlob(cleaned_text).sentiment.subjectivity

    # Construct feature vector
    X_input = np.zeros((1, len(feature_names)))
    embedding_dim = features.shape[1] if len(features.shape) > 1 else len(features)

    for i, feature in enumerate(feature_names):
        if feature.startswith("embedding_"):
            embedding_index = int(feature.split("_")[-1])
            if embedding_index < embedding_dim:
                X_input[0, i] = features[0, embedding_index]
        elif feature == "polarity":
            X_input[0, i] = polarity
        elif feature == "subjectivity":
            X_input[0, i] = subjectivity

    print("[DEBUG] Final Feature Vector for Model:", X_input)

    # Predict sockpuppet status
    probabilities = model.predict_proba(X_input)[0]
    print("[DEBUG] Prediction Probabilities:", probabilities)

    prediction = 1 if probabilities[1] >= 0.5 else 0
    result = "sockpuppet" if prediction == 1 else "non-sockpuppet"
    confidence = probabilities[1] * 100 if prediction == 1 else probabilities[0] * 100

    return render_template("result.html", prediction=result, confidence=f"{confidence:.2f}%")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
