# -*- coding: utf-8 -*-
"""
Diabetes Prediction System
Algorithm: Random Forest
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("diabetes.csv")

# ===============================
# 2. Separate Features & Target
# ===============================
X = df.drop(columns="Outcome", axis=1)
Y = df["Outcome"]

# ===============================
# 3. Train/Test Split
# ===============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X.values, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# ===============================
# 4. Train Random Forest Model
# ===============================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, Y_train)

# ===============================
# 5. Accuracy Check
# ===============================
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

print("\n🩺 Diabetes Model Performance")
print("=" * 50)
print(f"Training Accuracy : {train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {test_acc * 100:.2f}%")

# ===============================
# 6. Save Model
# ===============================
pickle.dump(model, open("diabetes_model.pkl", "wb"))

# ===============================
# 7. User Input System
# ===============================
print("\nEnter Patient Details:")

def num_input(msg):
    return float(input(msg))

preg = num_input("Pregnancies: ")
glucose = num_input("Glucose Level: ")
bp = num_input("Blood Pressure: ")
skin = num_input("Skin Thickness: ")
insulin = num_input("Insulin: ")
bmi = num_input("BMI: ")
dpf = num_input("Diabetes Pedigree Function: ")
age = num_input("Age: ")

input_data = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
input_array = np.asarray(input_data).reshape(1, -1)

# ===============================
# 8. Predict
# ===============================
prediction = model.predict(input_array)[0]
probability = model.predict_proba(input_array)[0][1] * 100

# ===============================
# 9. Output Result
# ===============================
print("\n📊 Prediction Result")
print("=" * 50)
print(f"Diabetes Probability    : {probability:.2f}%")
print(f"No Diabetes Probability : {100 - probability:.2f}%")

if prediction == 1:
    print("\n🟥 Result: Person HAS Diabetes")
else:
    print("\n🟩 Result: Person does NOT have Diabetes")

# ===============================
# 10. Personalized Suggestions
# ===============================
def generate_diabetes_suggestions(input_data, prob):
    suggestions = []
    glucose = input_data[1]
    bmi = input_data[5]
    age = input_data[7]

    if glucose > 125:
        suggestions.append("⚠️ High blood glucose: Consult doctor and monitor diet.")
    if bmi > 30:
        suggestions.append("⚖️ High BMI: Weight management recommended.")
    if age > 50:
        suggestions.append("👴 Age > 50: Regular check-ups advised.")

    if prob > 50:
        suggestions.append("⚠️ HIGH RISK: Immediate medical attention recommended.")
    else:
        suggestions.append("✅ LOW RISK: Maintain healthy lifestyle.")

    return suggestions

print("\n💡 Personalized Suggestions:")
suggestions = generate_diabetes_suggestions(input_data, probability)
for s in suggestions:
    print("- " + s)
