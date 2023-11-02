from django.urls import path
from api import views

app_name = 'api'

urlpatterns = [
    path('language-model/', views.language_model, name='language_model')
    ]
