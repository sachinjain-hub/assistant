import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Create folders if not exist
os.makedirs('model', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# Sample dataset
data = {
    'symptom': [
        'fever and cough',
        'headache and nausea',
        'joint pain and fatigue',
        'chest pain and breathlessness',
        'sneezing and runny nose',
        'stomach pain and diarrhea',
        'rash and itching',
        'weight loss and thirst',
        'fever and chills',
        'coughing blood',
        'shortness of breath and wheezing',
        'fatigue and yellow eyes',
        'frequent urination and back pain',
        'sore throat and fever',
        'eye redness and discharge',
        'swelling and redness',
        'blurry vision and headaches',
        'muscle weakness and tremors',
        'memory loss and confusion',
        'high fever and bleeding',
        'vomiting and abdominal pain',
        'cold feet and tiredness',
        'weight gain and fatigue',
        'numbness and tingling'
    ],
    'disease': [
        'flu',
        'migraine',
        'arthritis',
        'heart attack',
        'common cold',
        'food poisoning',
        'allergy',
        'diabetes',
        'malaria',
        'tuberculosis',
        'asthma',
        'hepatitis',
        'kidney infection',
        'strep throat',
        'conjunctivitis',
        'cellulitis',
        'glaucoma',
        'parkinson\'s disease',
        'alzheimer\'s disease',
        'dengue',
        'appendicitis',
        'anemia',
        'hypothyroidism',
        'neuropathy'
    ],
    'remedy': [
        'Rest and fluids',
        'Painkillers and rest',
        'Exercise and pain relief',
        'Immediate medical help',
        'Antihistamines and rest',
        'Hydration and rest',
        'Antihistamines and avoid allergen',
        'Insulin and diet control',
        'Antimalarial drugs',
        'Antibiotics and long-term care',
        'Inhaler and avoid triggers',
        'Rest and antiviral medication',
        'Antibiotics and hydration',
        'Antibiotics and warm fluids',
        'Eye drops and hygiene',
        'Antibiotics',
        'Eye drops and surgery if needed',
        'Medication and therapy',
        'Cognitive therapy and support',
        'Fluid replacement and monitoring',
        'Surgery',
        'Iron supplements and diet',
        'Thyroid hormone replacement',
        'Pain management and physiotherapy'
    ]
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv('dataset/symptoms_disease.csv', index=False)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['symptom'])
y = df['disease']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(model, open('model/disease_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("✅ Model and vectorizer saved successfully.")
