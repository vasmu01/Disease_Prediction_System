# -*- coding: utf-8 -*-
"""
Heart Disease Prediction System
Algorithm: Random Forest Only
(No StandardScaler used)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("heart.csv")

# Replace '?' with NaN and convert all columns to numeric
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# =========================
# FEATURES & TARGET
# =========================
FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']
TARGET = 'target'

X = df[FEATURES]
y = df[TARGET]

# =========================
# TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# TRAIN RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# =========================
# CHECK ACCURACY
# =========================
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n✅ Training Accuracy: {train_acc*100:.2f}%")
print(f"✅ Testing Accuracy : {test_acc*100:.2f}%")

# =========================
# SAVE MODEL
# =========================
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

# =========================
# USER INPUT & PREDICTION
# =========================
print("\n❤️ Heart Disease Prediction System")

user_input = {}
for f in FEATURES:
    while True:
        try:
            user_input[f] = float(input(f"{f}: "))
            break
        except ValueError:
            print("❌ Enter a valid number")

# Prepare input
input_array = np.array([list(user_input[f] for f in FEATURES)])

# Predict
prediction = model.predict(input_array)[0]
probability = model.predict_proba(input_array)[0][1] * 100

print(f"\n📊 Heart Disease Probability: {probability:.2f}%")
print("🟥 HAS Heart Disease" if prediction == 1 else "🟩 Does NOT have Heart Disease")

# =========================
# PERSONALIZED SUGGESTIONS
# =========================
age = user_input['age']
sex = user_input['sex']
trestbps = user_input['trestbps']
chol = user_input['chol']
fbs = user_input['fbs']
thalach = user_input['thalach']
exang = user_input['exang']
oldpeak = user_input['oldpeak']

suggestions = []

if age > 60:
    suggestions.append("👴 Age > 60: Regular heart check-ups recommended.")
elif age > 45:
    suggestions.append("🧑 Age 45–60: 150 min/week moderate exercise recommended.")

if sex == 1:
    suggestions.append("♂️ Male: Focus on cholesterol control.")
else:
    suggestions.append("♀️ Female: Maintain healthy heart habits.")

if trestbps > 140:
    suggestions.append("🩸 High BP: Reduce salt intake and manage stress.")
if chol > 200:
    suggestions.append("🍳 High cholesterol: Eat fiber-rich foods, avoid saturated fats.")
if fbs == 1:
    suggestions.append("🍬 High blood sugar: Control carbs, check diabetes regularly.")
if thalach < 140:
    suggestions.append("❤️ Low max heart rate: Increase aerobic exercise.")
if exang == 1:
    suggestions.append("🚭 Exercise-induced angina: Avoid heavy exertion and smoking.")
if oldpeak > 1:
    suggestions.append("📈 High ST depression: Consult doctor immediately.")

if probability > 50:
    suggestions.append("⚠️ HIGH RISK: Consult cardiologist urgently.")
else:
    suggestions.append("✅ LOW RISK: Maintain healthy lifestyle.")

print("\n💡 Personalized Suggestions:")
for s in suggestions:
    print("- " + s)
