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


model_path = os.path.join(settings.BASE_DIR, 'app', 'lung_model.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'app', 'lung_scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)


# Create your views here.

# views function for Lung Prediction :
def predict_lung_cancer(request):
    result = ""
    if request.method == "POST":
        gender = 1 if request.POST['gender'].upper() == 'M' else 2
        age = int(request.POST['age'])
        smoking = int(request.POST['smoking'])
        yellow_fingers = int(request.POST['yellow_fingers'])
        anxiety = int(request.POST['anxiety'])
        peer_pressure = int(request.POST['peer_pressure'])
        chronic_disease = int(request.POST['chronic_disease'])
        fatigue = int(request.POST['fatigue'])
        allergy = int(request.POST['allergy'])
        wheezing = int(request.POST['wheezing'])
        alcohol_consuming = int(request.POST['alcohol_consuming'])
        coughing = int(request.POST['coughing'])
        shortness_of_breath = int(request.POST['shortness_of_breath'])
        swallowing_difficulty = int(request.POST['swallowing_difficulty'])
        chest_pain = int(request.POST['chest_pain'])
        input_data = [
            gender, age, smoking, yellow_fingers, anxiety,
            peer_pressure, chronic_disease, fatigue, allergy,
            wheezing, alcohol_consuming, coughing,
            shortness_of_breath, swallowing_difficulty, chest_pain
        ]

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        
        result = "HAS Lung Cancer" if prediction[0]==1 else "Does NOT have Lung Cancer"

    return render(request, "index.html", {"result": result})




