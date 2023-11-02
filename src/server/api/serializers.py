from rest_framework import serializers
from .models import LanguageModel

class LanguageModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = LanguageModel
        fields = ['model_name', 'model_size', 'prompt', 'response']
        
