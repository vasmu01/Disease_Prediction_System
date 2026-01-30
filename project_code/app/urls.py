from django.contrib import admin
from django.urls import path, include
from app import views

urlpatterns = [
    path('', views.home, name='home'),
    path('heartprediction/', views.heart_prediction, name='heartprediction'),
    path('lungcancerprediction/', views.lungcancer_prediction, name='lungcancerprediction'),
    path('breastcancerprediction/', views.breastcancer_prediction, name='breastcancerprediction'),
    path('diabetiesprediction/', views.diabeties_prediction, name='diabetiesprediction'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.user_login, name='signin'),
    path('logout/', views.user_logout, name='logout'),
]
