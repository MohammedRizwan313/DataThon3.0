import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, render_template, request, jsonify
import joblib
from sentiment import app

"""
label_encoder = joblib.load('label_encoder.pkl')
#model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
model_path=r"C:\\Users\\moham\\Desktop\\ML deploy\\bert_emotion_model.pth"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(torch.load('bert_emotion_model.pth'))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
"""
label_encoder = joblib.load('label_encoder.pkl')
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
model_path = r"C:\\Users\\moham\\Desktop\\ML deploy\\bert_emotion_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Load the label encoder


@app.route('/', methods=['GET', 'POST'])
def home_page():
    predicted_emotion = None
    
    if request.method == 'POST':
        input_text = request.form.get('url')
        inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    
    return render_template('index.html', predicted_emotion=predicted_emotion)



