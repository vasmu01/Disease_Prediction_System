from django.contrib import admin
from django.urls import path, include
from app import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.user_login, name='signin'),
    path('logout/', views.user_logout, name='logout'),
]
