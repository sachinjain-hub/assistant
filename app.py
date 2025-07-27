from flask import Flask, render_template, request, jsonify
from utils.text_to_speech import text_to_speech
from utils.hospital_locator import find_nearest_hospitals
import pickle
import pandas as pd
import os

# Load model and vectorizer
model = pickle.load(open('model/disease_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptom = request.form['symptom']
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    symptom_vec = vectorizer.transform([symptom])
    disease = model.predict(symptom_vec)[0]
    remedy = get_remedy(disease)

    if lat and lon:
        hospitals = find_nearest_hospitals(lat, lon)
    else:
        hospitals = "⚠️ Location not provided. Cannot fetch hospitals."

    text_to_speech(f"The predicted disease is {disease}. Remedy: {remedy}.")

    return jsonify({
        'disease': disease,
        'remedy': remedy,
        'hospitals': hospitals
    })

def get_remedy(disease):
    df = pd.read_csv('dataset/symptoms_disease.csv')
    remedy = df.loc[df['disease'] == disease, 'remedy'].values[0]
    return remedy

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
