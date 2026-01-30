"""
Breast Cancer Prediction System
Algorithm: Random Forest
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("breast-cancer.csv")  # Replace with your CSV path

# ===============================
# 2. Encode Target & Drop ID
# ===============================
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
df.drop(columns='id', inplace=True)

# ===============================
# 3. Separate Features & Target
# ===============================
X = df.drop(columns='diagnosis')
Y = df['diagnosis']

# ===============================
# 4. Train/Test Split
# ===============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X.values, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# ===============================
# 5. Train Random Forest Model
# ===============================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, Y_train)

# ===============================
# 6. Accuracy Check
# ===============================
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

print("\n🎀 Breast Cancer Model Performance")
print("=" * 50)
print(f"Training Accuracy : {train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {test_acc * 100:.2f}%")

# ===============================
# 7. Save Model
# ===============================
pickle.dump(model, open("breast_model.pkl", "wb"))

# ===============================
# 8. User Input System
# ===============================
print("\nEnter Patient Tumor Details:")

def num_input(msg):
    return float(input(msg))

features = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
    'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
]

user_data = [num_input(f"{f}: ") for f in features]

input_array = np.asarray(user_data).reshape(1, -1)

# ===============================
# 9. Predict
# ===============================
prediction = model.predict(input_array)[0]
probability = model.predict_proba(input_array)[0][1] * 100

# ===============================
# 10. Output Result
# ===============================
print("\n📊 Prediction Result")
print("=" * 50)
print(f"Cancer Probability     : {probability:.2f}%")
print(f"No Cancer Probability  : {100 - probability:.2f}%")

if prediction == 1:
    print("\n🟥 Result: MALIGNANT (Breast Cancer Detected)")
else:
    print("\n🟩 Result: BENIGN (No Breast Cancer)")

# ===============================
# 11. Personalized Suggestions
# ===============================
def generate_breast_cancer_suggestions(input_data, prob):
    suggestions = []

    radius_mean = input_data[0]
    area_mean = input_data[3]
    
    # Tumor size based suggestions
    if radius_mean > 15:
        suggestions.append("⚠️ Large tumor size suspected: Consult oncologist immediately.")
    if area_mean > 500:
        suggestions.append("📏 Large tumor area detected: Early intervention advised.")

    # Risk-based suggestion
    if prob > 50:
        suggestions.append("⚠️ HIGH RISK: Immediate medical consultation recommended.")
    else:
        suggestions.append("✅ LOW RISK: Continue regular screening and healthy lifestyle.")

    return suggestions

print("\n💡 Personalized Suggestions:")
suggestions = generate_breast_cancer_suggestions(user_data, probability)
for s in suggestions:
    print("- " + s)
