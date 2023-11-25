import pickle
import torch
import json
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

app = Flask(__name__)
app.json_encoder = json.JSONEncoder

model_dir = Path("models")  # Adjust the path if needed
with (model_dir / "code2idx.pkl").open("rb") as f:
    code2idx = pickle.load(f)

model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Make prediction
    with torch.no_grad():
        output = model(**inputs)

    # Get predicted label
    prediction = output.logits.argmax(-1).item()

    # Map the label to the corresponding code
    code = [c for c, i in code2idx.items() if i == prediction][0]

    return jsonify({"predicted_codes": [code]})


if __name__ == "__main__":
    app.run(debug=False)
