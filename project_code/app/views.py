from django.shortcuts import render,redirect
import pandas as pd 
import pickle
import numpy as np
import os
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib import messages 
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

def home(request):
  return render(request,'home.html')

def signup(request):
  return render(request,'signup.html')

def signin(request):
  return render(request,'signin.html')

def heartp(request):
  return render(request,'heartp.html')


def lungcancerprediction(request):
  return render(request,'lungcancerprediction.html')


def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect('signup')

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
        )
        user.save()
        messages.success(request, "Account created successfully")
        return redirect('signin')

    return render(request, 'signup.html')

def user_login(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        if not email or not password:
            messages.error(request, "All fields are required")
            return redirect('signin')

        try:
            user_obj = User.objects.get(email=email)
        except User.DoesNotExist:
            messages.error(request, "Invalid email or password")
            return redirect('signin')

        # Authenticate using the username of the user object
        user = authenticate(
            request,
            username=user_obj.username,
            password=password
        )

        if user is not None:
            login(request, user)  # <-- Django's login
            messages.success(request, "Login successful")
            return redirect('home')
        else:
            messages.error(request, "Invalid email or password")
            return redirect('signin')

    return render(request, 'signin.html')


@login_required
def dashboard(request):
    return render(request, 'home.html')

def user_logout(request):
    logout(request)
    messages.success(request, "You have successfully logged out.")
    return redirect('signin')


# HEART DISEASE PREDICTION

MODEL_PATH = os.path.join(settings.BASE_DIR, "app", "models", "heart_model.pkl")

# Load the model safely
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
# List of features
FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']

# Realistic ranges for input validation
RANGES = {
    'age': (20, 90),
    'sex': (0, 2),  # 0: female, 1: male, 2: custom
    'cp': (0, 3),
    'trestbps': (90, 200),
    'chol': (100, 400),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'thalach': (70, 210),
    'exang': (0, 1),
    'oldpeak': (0, 6),
    'slope': (0, 2),
    'ca': (0, 4),
    'thal': (0, 3)
}

def heart_prediction(request):
    context = {}
    if request.method == "POST":
        # Get user input
        user_input = {}
        errors = []

        for f in FEATURES:
            value = request.POST.get(f)
            if value is None or value.strip() == "":
                errors.append(f"{f} is required.")
                continue
            try:
                val = float(value)
            except ValueError:
                errors.append(f"{f} must be a number.")
                continue

            # Validate realistic range
            min_val, max_val = RANGES[f]
            if val < min_val or val > max_val:
                errors.append(f"{f} must be between {min_val} and {max_val}.")
                continue

            user_input[f] = val

        if errors:
            context['errors'] = errors
        else:
            # Prepare input for prediction
            input_array = np.array([list(user_input.values())])
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0][1] * 100

            # Personalized suggestions
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

            context['prediction'] = prediction
            context['probability'] = round(probability, 2)
            context['suggestions'] = suggestions
            context['user_input'] = user_input

    return render(request, "heartprediction.html", context)





# LUNGCANCER PREDICTION
MODEL_PATH = os.path.join(settings.BASE_DIR, "app", "models", "lung_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Features
FEATURES = [
    'gender','age','smoking','yellow_fingers','anxiety','peer_pressure',
    'chronic_disease','fatigue','allergy','wheezing','alcohol','coughing',
    'shortness_breath','swallowing','chest_pain'
]

# Realistic ranges for validation
RANGES = {
    'gender': (0,1),
    'age': (1,120),
    'smoking': (0,1),
    'yellow_fingers': (0,1),
    'anxiety': (0,1),
    'peer_pressure': (0,1),
    'chronic_disease': (0,1),
    'fatigue': (0,1),
    'allergy': (0,1),
    'wheezing': (0,1),
    'alcohol': (0,1),
    'coughing': (0,1),
    'shortness_breath': (0,1),
    'swallowing': (0,1),
    'chest_pain': (0,1)
}

def lungcancer_prediction(request):
    context = {}
    if request.method == "POST":
        user_input = {}
        errors = []

        for f in FEATURES:
            value = request.POST.get(f)
            if value is None or value.strip() == "":
                errors.append(f"{f.replace('_',' ').title()} is required.")
                continue
            try:
                val = float(value)
            except ValueError:
                errors.append(f"{f.replace('_',' ').title()} must be a number.")
                continue

            min_val, max_val = RANGES[f]
            if val < min_val or val > max_val:
                errors.append(f"{f.replace('_',' ').title()} must be between {min_val} and {max_val}.")
                continue

            user_input[f] = val

        if errors:
            context['errors'] = errors
        else:
            input_array = np.array([list(user_input.values())])
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0][1] * 100

            # Personalized suggestions
            age = user_input['age']
            gender = user_input['gender']
            fatigue = user_input['fatigue']
            chronic = user_input['chronic_disease']
            chest_pain = user_input['chest_pain']

            suggestions = []
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

            if probability > 50:
                suggestions.append("⚠️ HIGH RISK: Consult doctor urgently.")
            else:
                suggestions.append("✅ LOW RISK: Maintain healthy lifestyle.")

            context['prediction'] = prediction
            context['probability'] = round(probability, 2)
            context['suggestions'] = suggestions
            context['user_input'] = user_input

    return render(request, "lungcancerprediction.html", context)




# BREAST CANCER PREDICTION

MODEL_PATH = os.path.join(settings.BASE_DIR, "app", "models", "breast_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Features
FEATURES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
    'compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'
]

# Realistic ranges (example; you can tune based on dataset stats)
RANGES = {f: (0, 50) for f in FEATURES}  # Most features are continuous numeric values

def breast_prediction(request):
    context = {}
    if request.method == "POST":
        user_input = {}
        errors = []

        for f in FEATURES:
            value = request.POST.get(f)
            if value is None or value.strip() == "":
                errors.append(f"{f.replace('_',' ').title()} is required.")
                continue
            try:
                val = float(value)
            except ValueError:
                errors.append(f"{f.replace('_',' ').title()} must be a number.")
                continue

            min_val, max_val = RANGES[f]
            if val < min_val or val > max_val:
                errors.append(f"{f.replace('_',' ').title()} must be between {min_val} and {max_val}.")
                continue

            user_input[f] = val

        if errors:
            context['errors'] = errors
        else:
            input_array = np.array([list(user_input.values())])
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0][1] * 100

            # Suggestions
            radius_mean = user_input['radius_mean']
            area_mean = user_input['area_mean']

            suggestions = []
            if radius_mean > 15:
                suggestions.append("⚠️ Large tumor size suspected: Consult oncologist immediately.")
            if area_mean > 500:
                suggestions.append("📏 Large tumor area detected: Early intervention advised.")

            if probability > 50:
                suggestions.append("⚠️ HIGH RISK: Immediate medical consultation recommended.")
            else:
                suggestions.append("✅ LOW RISK: Continue regular screening and healthy lifestyle.")

            context['prediction'] = prediction
            context['probability'] = round(probability, 2)
            context['suggestions'] = suggestions
            context['user_input'] = user_input

    return render(request, "breastcancerprediction.html", context)



# DIABETIES PREDICTION
MODEL_PATH = os.path.join(settings.BASE_DIR, "app", "models", "diabetes_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Diabetes model not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===============================
# Personalized Suggestions Function
# ===============================
def generate_diabetes_suggestions(input_data, prob):
    suggestions = []
    pregnancies, glucose, bp, skin, insulin, bmi, dpf, age = input_data

    # Glucose
    if glucose > 125:
        suggestions.append("⚠️ High blood glucose: Consult doctor and monitor diet.")
    elif glucose < 70:
        suggestions.append("⚠️ Low blood glucose: Monitor diet and consult doctor if needed.")

    # BMI
    if bmi > 30:
        suggestions.append("⚖️ High BMI: Weight management recommended.")
    elif bmi < 18.5:
        suggestions.append("⚖️ Low BMI: Ensure proper nutrition.")

    # Age
    if age > 50:
        suggestions.append("👴 Age > 50: Regular check-ups advised.")
    elif age < 18:
        suggestions.append("🧒 Age < 18: Monitor growth and health regularly.")

    # Pregnancies
    if pregnancies > 3:
        suggestions.append(f"🤰 High number of pregnancies ({pregnancies}): Monitor health carefully.")

    # Blood Pressure
    if bp > 130:
        suggestions.append("💓 High blood pressure: Regular check-ups and lifestyle adjustments recommended.")
    elif bp < 80:
        suggestions.append("💓 Low blood pressure: Ensure hydration and consult doctor if symptomatic.")

    # Insulin
    if insulin > 200:
        suggestions.append("💉 High insulin levels: Consult doctor about insulin resistance.")
    elif insulin < 30:
        suggestions.append("💉 Low insulin levels: Monitor glucose and diet.")

    # Diabetes Pedigree Function
    if dpf > 1:
        suggestions.append("📊 High diabetes pedigree score: Increased risk, regular monitoring recommended.")

    # Overall probability
    if prob > 50:
        suggestions.append("⚠️ HIGH RISK: Immediate medical attention recommended.")
    else:
        suggestions.append("✅ LOW RISK: Maintain healthy lifestyle and regular check-ups.")

    return suggestions

# ===============================
# Diabetes Prediction View
# ===============================
def diabeties_prediction(request):
    context = {}

    if request.method == "POST":
        user_input = {}
        errors = []

        # Fields expected from the form
        fields = ["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "dpf", "age"]

        # Validate inputs (only numbers allowed)
        for f in fields:
            value = request.POST.get(f)
            if value is None or value.strip() == "":
                errors.append(f"{f.replace('_',' ').title()} is required.")
                continue
            try:
                val = float(value)
            except ValueError:
                errors.append(f"{f.replace('_',' ').title()} must be a number.")
                continue
            user_input[f] = val

        if errors:
            context["errors"] = errors
            context["user_input"] = user_input
        else:
            # Make prediction
            input_array = np.array([list(user_input.values())])
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0][1] * 100

            # Generate suggestions
            suggestions = generate_diabetes_suggestions(list(user_input.values()), probability)

            # Pass all to template
            context.update({
                "user_input": user_input,
                "prediction": "HAS Diabetes" if prediction == 1 else "Does NOT have Diabetes",
                "probability": round(probability, 2),
                "suggestions": suggestions
            })

    return render(request, "diabetiesprediction.html", context)
