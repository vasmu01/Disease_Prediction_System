import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. Load dataset
# ===============================
dataset = pd.read_csv("lungcancer.csv")  # Replace with your CSV path

# ===============================
# 2. Map categorical values to numeric
# ===============================
mapping = {
    'YES': 1,
    'NO': 0,
    'M': 1,
    'F': 0
}

for col in dataset.columns:
    if dataset[col].dtype == object:
        dataset[col] = dataset[col].map(mapping)

# ===============================
# 3. Remove rows with missing values
# ===============================
dataset.dropna(inplace=True)

# ===============================
# 4. Separate features & target
# ===============================
X = dataset.drop(columns='Outcome')
Y = dataset['Outcome']

# ===============================
# 5. Train/Test Split
# ===============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X.values, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# ===============================
# 6. Train RandomForest Model
# ===============================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, Y_train)

# ===============================
# 7. Accuracy Check
# ===============================
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

print("\n🫁 Lung Cancer Model Performance")
print("=" * 50)
print(f"Training Accuracy : {train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {test_acc * 100:.2f}%")


# ===============================
# 9. Predictive System (User Input)
# ===============================
print("\nEnter Patient Details (0 = NO, 1 = YES unless stated, M=1, F=0):")

def num_input(msg):
    return float(input(msg))

gender = num_input("Gender (M=1, F=0): ")
age = num_input("Age: ")
smoking = num_input("Smoking: ")
yellow_fingers = num_input("Yellow Fingers: ")
anxiety = num_input("Anxiety: ")
peer_pressure = num_input("Peer Pressure: ")
chronic_disease = num_input("Chronic Disease: ")
fatigue = num_input("Fatigue: ")
allergy = num_input("Allergy: ")
wheezing = num_input("Wheezing: ")
alcohol = num_input("Alcohol Consuming: ")
coughing = num_input("Coughing: ")
shortness_breath = num_input("Shortness of Breath: ")
swallowing = num_input("Swallowing Difficulty: ")
chest_pain = num_input("Chest Pain: ")

# ===============================
# 10. Prepare Input & Predict
# ===============================
input_data = [
    gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
    chronic_disease, fatigue, allergy, wheezing,
    alcohol, coughing, shortness_breath,
    swallowing, chest_pain
]

input_array = np.asarray(input_data).reshape(1, -1)

# Prediction
prediction = model.predict(input_array)
probability = model.predict_proba(input_array)[0][1] * 100

# ===============================
# 11. Output Result
# ===============================
print("\n📊 Prediction Result")
print("=" * 50)
print(f"Disease Probability : {probability:.2f}%")
print(f"No Disease Probability : {100 - probability:.2f}%")

if prediction[0] == 1:
    print("\n🟥 Result: Person HAS Disease")
else:
    print("\n🟩 Result: Person does NOT have Disease")

# ===============================
# 12. Optional: Personalized Suggestions
# ===============================
def generate_suggestions(input_data, prob):
    suggestions = []
    age = input_data[1]
    gender = input_data[0]
    fatigue = input_data[7]
    chronic = input_data[6]
    chest_pain = input_data[14]

    if age > 60:
        suggestions.append("👴 Age > 60: Regular checkups recommended.")
    if gender == 1:
        suggestions.append("♂️ Male: Monitor lifestyle risk factors.")
    else:
        suggestions.append("♀️ Female: Maintain healthy habits.")

    if fatigue == 1:
        suggestions.append("💤 Fatigue: Ensure proper rest and check underlying conditions.")
    if chronic == 1:
        suggestions.append("🩺 Chronic Disease: Follow prescribed treatments.")
    if chest_pain == 1:
        suggestions.append("❤️ Chest Pain: Immediate doctor consultation advised.")

    if prob > 50:
        suggestions.append("⚠️ HIGH RISK: Consult doctor urgently.")
    else:
        suggestions.append("✅ LOW RISK: Maintain healthy lifestyle.")

    return suggestions

print("\n💡 Personalized Suggestions:")
suggestions = generate_suggestions(input_data, probability)
for s in suggestions:
    print("- " + s)

 # Save the trained model
with open("lungcancer_model.pkl", "wb") as f:
    pickle.dump(model, f)
