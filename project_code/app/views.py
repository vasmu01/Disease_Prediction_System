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


# Load the trained model (only once when server starts)

model = os.path.join(settings.BASE_DIR, "app", "models", "lung_model.pkl")

if not os.path.exists(model):
    raise FileNotFoundError(f"Diabetes model not found at {model}")

with open(model, "rb") as f:
    model = pickle.load(f)
    
def lungcancer_prediction(request):
    if request.method == 'POST':
        try:
            # Collect all 15 inputs from form
            gender = int(request.POST.get('gender'))
            age = float(request.POST.get('age'))
            smoking = int(request.POST.get('smoking'))
            yellow_fingers = int(request.POST.get('yellow_fingers'))
            anxiety = int(request.POST.get('anxiety'))
            peer_pressure = int(request.POST.get('peer_pressure'))
            chronic_disease = int(request.POST.get('chronic_disease'))
            fatigue = int(request.POST.get('fatigue'))
            allergy = int(request.POST.get('allergy'))
            wheezing = int(request.POST.get('wheezing'))
            alcohol = int(request.POST.get('alcohol'))
            coughing = int(request.POST.get('coughing'))
            shortness_breath = int(request.POST.get('shortness_breath'))
            swallowing = int(request.POST.get('swallowing'))
            chest_pain = int(request.POST.get('chest_pain'))

            # Prepare input for model
            input_data = [
                gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                chronic_disease, fatigue, allergy, wheezing,
                alcohol, coughing, shortness_breath,
                swallowing, chest_pain
            ]

            input_array = np.array([input_data])

            # Make prediction
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0][1] * 100

            # Prepare result
            context = {
                'probability': round(probability, 2),
                'no_disease_prob': round(100 - probability, 2),
                'has_disease': prediction == 1,
            }

            return render(request, 'lungcancerprediction.html', context)

        except (ValueError, TypeError):
            # If any input is invalid
            return render(request, 'lungcancerprediction.html', {'error': 'कृपया सबै जानकारी सही तरिकाले भर्नुहोस्।'})

    # GET request → show empty form
    return render(request, 'lungcancerprediction.html')




# =========================
# Load Heart Model
# =========================
model_path = os.path.join(settings.BASE_DIR, "app", 'models', 'heart_model.pkl')
heart_model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        heart_model = pickle.load(f)
else:
    print(f"Heart model not found at {model_path}")

# =========================
# Heart Prediction View
# =========================
def heart_prediction(request):
    context = {
        'probability': None,
        'no_disease_prob': None,
        'has_disease': None,
        'error': None
    }

    if request.method == 'POST':
        try:
            # Collect all 13 features from form
            features = [
                float(request.POST.get('age', 0)),
                int(request.POST.get('sex', 0)),
                int(request.POST.get('cp', 0)),
                float(request.POST.get('trestbps', 0)),
                float(request.POST.get('chol', 0)),
                int(request.POST.get('fbs', 0)),
                int(request.POST.get('restecg', 0)),
                float(request.POST.get('thalach', 0)),
                int(request.POST.get('exang', 0)),
                float(request.POST.get('oldpeak', 0)),
                int(request.POST.get('slope', 0)),
                float(request.POST.get('ca', 0)),
                int(request.POST.get('thal', 0)),
            ]

            input_array = np.array([features])

            if heart_model is None:
                context['error'] = "Heart model file not loaded."
            else:
                prediction = heart_model.predict(input_array)[0]
                probability = heart_model.predict_proba(input_array)[0][1] * 100

                context.update({
                    'probability': round(probability, 2),
                    'no_disease_prob': round(100 - probability, 2),
                    'has_disease': prediction == 1,
                })

        except (ValueError, TypeError):
            context['error'] = "कृपया सबै फिल्डहरू सही तरिकाले भर्नुहोस् (संख्या मात्र)।"

    return render(request, 'heartprediction.html', context)
