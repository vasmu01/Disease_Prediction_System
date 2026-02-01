"""
Health Prediction Web Application - Django Views
Handles user authentication, dashboard, and ML model predictions for various diseases
"""

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
import pandas as pd
import pickle
import numpy as np
import os


# =============================================================================
# AUTHENTICATION VIEWS
# =============================================================================

def home(request):
    """Render the home page."""
    return render(request, 'home.html')


def signup(request):
    """Handle user registration."""
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect('signup')

        # Create new user
        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        messages.success(request, "Account created successfully")
        return redirect('signin')
    
    return render(request, 'signup.html')


def signin(request):
    """Render signin page."""
    return render(request, 'signin.html')


def user_login(request):
    """Handle user login with email."""
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

        # Authenticate using username (Django requirement)
        user = authenticate(request, username=user_obj.username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful")
            return redirect('home')
        else:
            messages.error(request, "Invalid email or password")
            return redirect('signin')

    return render(request, 'signin.html')


@login_required
def dashboard(request):
    """Protected dashboard view (requires login)."""
    return render(request, 'home.html')


def user_logout(request):
    """Handle user logout."""
    logout(request)
    messages.success(request, "You have successfully logged out.")
    return redirect('signin')


# =============================================================================
# MODEL INITIALIZATION (Global - Loaded once at startup)
# =============================================================================

# Diabetes Model
DIABETES_MODEL_PATH = os.path.join(settings.BASE_DIR, "app", "models", "diabetes_model.pkl")
diabetes_model = None
if os.path.exists(DIABETES_MODEL_PATH):
    with open(DIABETES_MODEL_PATH, "rb") as f:
        diabetes_model = pickle.load(f)
else:
    print(f"Diabetes model not found at {DIABETES_MODEL_PATH}")


# Lung Cancer Model
LUNG_MODEL_PATH = os.path.join(settings.BASE_DIR, "app", "models", "lung_model.pkl")
lung_model = None
if os.path.exists(LUNG_MODEL_PATH):
    with open(LUNG_MODEL_PATH, "rb") as f:
        lung_model = pickle.load(f)
else:
    print(f"Lung model not found at {LUNG_MODEL_PATH}")


# Heart Disease Model
HEART_MODEL_PATH = os.path.join(settings.BASE_DIR, "app", "models", "heart_model.pkl")
heart_model = None
if os.path.exists(HEART_MODEL_PATH):
    with open(HEART_MODEL_PATH, "rb") as f:
        heart_model = pickle.load(f)
else:
    print(f"Heart model not found at {HEART_MODEL_PATH}")


# Breast Cancer Model
BREAST_MODEL_PATH = os.path.join(settings.BASE_DIR, "app", "models", "breast_model.pkl")
breast_model = None
if os.path.exists(BREAST_MODEL_PATH):
    with open(BREAST_MODEL_PATH, "rb") as f:
        breast_model = pickle.load(f)
else:
    print(f"Breast model not found at {BREAST_MODEL_PATH}")


# =============================================================================
# DIABETES PREDICTION
# =============================================================================


def diabeties_prediction(request):
    """
    Diabetes prediction view with personalized health suggestions.
    """
    context = {}
    fields = ["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "dpf", "age"]

    if request.method == "POST":
        user_input = {}
        errors = []

        # Validate all required fields
        for field in fields:
            value = request.POST.get(field)
            if not value or value.strip() == "":
                errors.append(f"{field.replace('_', ' ').title()} is required.")
                continue
            try:
                user_input[field] = float(value)
            except ValueError:
                errors.append(f"{field.replace('_', ' ').title()} must be a number.")

        context["user_input"] = user_input

        # Return validation errors if any
        if errors:
            context["errors"] = errors
        # Check if model is loaded
        elif diabetes_model is None:
            context["errors"] = ["Diabetes model not loaded."]
        else:
            # Make prediction
            input_array = np.array([list(user_input.values())])
            prediction = diabetes_model.predict(input_array)[0]
            probability = diabetes_model.predict_proba(input_array)[0][1] * 100
            probability = round(probability, 2)  # Round for display

            # Calculate risk remaining
            risk_remaining = round(100 - probability, 2)

            # Generate suggestions (pass user_input dict and probability)

            context.update({
                "prediction": "HAS Diabetes" if prediction == 1 else "Does NOT have Diabetes",
                "probability": probability,
                "risk_remaining": risk_remaining,
            })

    return render(request, "diabetiesprediction.html", context)

# =============================================================================
# LUNG CANCER PREDICTION
# =============================================================================

def lungcancer_prediction(request):
    """Lung cancer risk prediction using 15 features."""
    if request.method == 'POST':
        try:
            # Collect 15 input features
            features = [
                int(request.POST.get('gender')),
                float(request.POST.get('age')),
                int(request.POST.get('smoking')),
                int(request.POST.get('yellow_fingers')),
                int(request.POST.get('anxiety')),
                int(request.POST.get('peer_pressure')),
                int(request.POST.get('chronic_disease')),
                int(request.POST.get('fatigue')),
                int(request.POST.get('allergy')),
                int(request.POST.get('wheezing')),
                int(request.POST.get('alcohol')),
                int(request.POST.get('coughing')),
                int(request.POST.get('shortness_breath')),
                int(request.POST.get('swallowing')),
                int(request.POST.get('chest_pain'))
            ]

            input_array = np.array([features])

            if lung_model is None:
                return render(request, 'lungcancerprediction.html', 
                            {'error': 'Lung cancer model not loaded.'})

            # Make prediction
            prediction = lung_model.predict(input_array)[0]
            probability = lung_model.predict_proba(input_array)[0][1] * 100

            context = {
                'probability': round(probability, 2),
                'no_disease_prob': round(100 - probability, 2),
                'has_disease': prediction == 1,
            }
            return render(request, 'lungcancerprediction.html', context)

        except (ValueError, TypeError):
            return render(request, 'lungcancerprediction.html', 
                         {'error': 'कृपया सबै जानकारी सही तरिकाले भर्नुहोस्।'})

    return render(request, 'lungcancerprediction.html')


# =============================================================================
# HEART DISEASE PREDICTION
# =============================================================================

def heart_prediction(request):
    """Heart disease prediction using 13 clinical features."""
    context = {
        'probability': None,
        'no_disease_prob': None,
        'has_disease': None,
        'error': None
    }

    if request.method == 'POST':
        try:
            # 13 heart disease features in exact training order
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


# =============================================================================
# BREAST CANCER PREDICTION
# =============================================================================

def breastcancer_prediction(request):
    """Breast cancer prediction using 30 tumor features."""
    context = {
        'probability': None,
        'no_cancer_prob': None,
        'is_malignant': False,
        'error': None,
    }

    if request.method == 'POST':
        try:
            if breast_model is None:
                context['error'] = "Breast cancer model not loaded."
                return render(request, 'breastcancerprediction.html', context)

            # 30 breast cancer features (mean, se, worst) in exact training order
            features = [
                # Mean features
                float(request.POST.get('radius_mean')),
                float(request.POST.get('texture_mean')),
                float(request.POST.get('perimeter_mean')),
                float(request.POST.get('area_mean')),
                float(request.POST.get('smoothness_mean')),
                float(request.POST.get('compactness_mean')),
                float(request.POST.get('concavity_mean')),
                float(request.POST.get('concave_points_mean')),
                float(request.POST.get('symmetry_mean')),
                float(request.POST.get('fractal_dimension_mean')),
                
                # SE features
                float(request.POST.get('radius_se')),
                float(request.POST.get('texture_se')),
                float(request.POST.get('perimeter_se')),
                float(request.POST.get('area_se')),
                float(request.POST.get('smoothness_se')),
                float(request.POST.get('compactness_se')),
                float(request.POST.get('concavity_se')),
                float(request.POST.get('concave_points_se')),
                float(request.POST.get('symmetry_se')),
                float(request.POST.get('fractal_dimension_se')),
                
                # Worst features
                float(request.POST.get('radius_worst')),
                float(request.POST.get('texture_worst')),
                float(request.POST.get('perimeter_worst')),
                float(request.POST.get('area_worst')),
                float(request.POST.get('smoothness_worst')),
                float(request.POST.get('compactness_worst')),
                float(request.POST.get('concavity_worst')),
                float(request.POST.get('concave_points_worst')),
                float(request.POST.get('symmetry_worst')),
                float(request.POST.get('fractal_dimension_worst')),
            ]

            input_array = np.array([features])
            prediction = breast_model.predict(input_array)[0]
            probability = breast_model.predict_proba(input_array)[0][1] * 100

            context.update({
                'probability': round(probability, 2),
                'no_cancer_prob': round(100 - probability, 2),
                'is_malignant': prediction == 1,
            })

        except (ValueError, TypeError):
            context['error'] = "Please enter valid numeric values for all fields."
        except Exception as e:
            context['error'] = f"Prediction error: {str(e)}"

    return render(request, 'breastcancerprediction.html', context)


# =============================================================================
# SIMPLE PAGES (Prediction pages without logic)
# =============================================================================

def heartp(request):
    """Render heart prediction page."""
    return render(request, 'heartp.html')


def lungcancerprediction(request):
    """Render lung cancer prediction page."""
    return render(request, 'lungcancerprediction.html')
