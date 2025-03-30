import pandas as pd
import numpy as np
import joblib
import os
import webbrowser
from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from threading import Timer

# Specify the correct template folder
app = Flask(__name__, template_folder="html")

# Load trained model and metadata
model_path = "model/random_forest.pkl"
feature_metadata_path = "model/feature_metadata.pkl"
embeddings_lookup_path = "model/embeddings_lookup.csv"

print("[INFO] Loading model...")
model = joblib.load(model_path)
print("[INFO] Model loaded successfully.")

print("[INFO] Loading feature metadata...")
feature_metadata = joblib.load(feature_metadata_path)
feature_names = feature_metadata['features']
print("[INFO] Feature metadata loaded.")

# Load embeddings lookup table
if os.path.exists(embeddings_lookup_path):
    print("[INFO] Loading embeddings lookup table...")
    embeddings_lookup = pd.read_csv(embeddings_lookup_path)
    embeddings_lookup.set_index('edit_text', inplace=True)
    print("[INFO] Embeddings lookup table loaded.")
else:
    print("[WARNING] Embeddings lookup table not found. Defaulting to dynamic embedding generation.")
    embeddings_lookup = None

# Load Sentence-BERT model for dynamic embeddings
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json if request.is_json else request.form
    edit_text = data.get('input_text', "").strip()

    if not edit_text:
        return jsonify({"error": "Missing 'input_text' in request."}), 400

    # Retrieve embeddings from lookup table
    if embeddings_lookup is not None and edit_text in embeddings_lookup.index:
        print("[INFO] Using precomputed embeddings.")
        features = embeddings_lookup.loc[edit_text].values.reshape(1, -1)
    else:
        print(f"[INFO] Generating new embeddings for: {edit_text}")
        features = sbert_model.encode([edit_text])
        features = features / np.linalg.norm(features)  # Normalize the new embeddings

    print("[DEBUG] Generated Embeddings Shape:", features.shape)
    print("[DEBUG] Generated Embeddings:", features)

    # Compute sentiment features
    polarity = TextBlob(edit_text).sentiment.polarity
    subjectivity = TextBlob(edit_text).sentiment.subjectivity

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
    
    return render_template("result.html", prediction=result)


def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)