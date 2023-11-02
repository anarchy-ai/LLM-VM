from django.db import models

class LanguageModel(models.Model):
    model_name = models.CharField(max_length=200)
    model_size = models.CharField(max_length=200)
    prompt = models.TextField()
    response = models.TextField()

    class Meta:
        ordering = ['id']

    def __str__(self):
        return f'The type of model is {self.model_name} and its size is {self.model_size} and the prompt is {self.prompt} with reponse: {self.response}.'
