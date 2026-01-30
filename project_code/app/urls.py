from django.contrib import admin
from django.urls import path, include
from app import views

urlpatterns = [
    path('', views.home, name='home'),
    path('heartprediction/', views.heart_prediction, name='heart_prediction'),
    path('lungcancerprediction/', views.lungcancer_prediction, name='lungcancer_prediction'),
    path('breastcancerprediction/', views.breast_prediction, name='breast_prediction'),
    path('diabetiesprediction/', views.diabeties_prediction, name='diabeties_prediction'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.user_login, name='signin'),
    path('logout/', views.user_logout, name='logout'),
]
