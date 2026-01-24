from django.db import models

# Create your models here.
class Data(models.Model):
    name = models.CharField(max_length=50)
    password = models.CharField(max_length=50 ,null=True, blank=True)
    cpassword = models.CharField(max_length=50 ,null=True, blank=True)
    email =  models.EmailField( max_length=254)
