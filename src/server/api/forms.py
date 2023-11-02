from django import forms

class LanguageModelForm(forms.Form):
    model_name = forms.CharField()
    model_size = forms.CharField()
    prompt = forms.CharField()
