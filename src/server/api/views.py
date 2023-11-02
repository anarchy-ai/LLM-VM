from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from .forms import LanguageModelForm
from .serializers import LanguageModelSerializer
from .models import LanguageModel
from llm_vm.client import Client

@api_view(['POST'])
@csrf_exempt
def language_model(request):
    form = LanguageModelForm(request.POST)
    if form.is_valid():
        cd = form.cleaned_data
        prompt = cd['prompt']
        model_size = cd['model_size']
        model_name = cd['model_name']

        if model_size == 'big_model':
            client = Client(big_model = model_name)
            response = client.complete(prompt = prompt, context = '')
            data_model = LanguageModel(model_name=model_name, model_size=model_size, prompt=prompt, response=response)
            serializer = LanguageModelSerializer(data_model)

            return Response(serializer.data)
        else:
            client = Client(small_model = model_name)
            response = client.complete(prompt = prompt, context = '')
            data_model = LanguageModel(model_name=model_name, model_size=model_size, prompt=prompt, response=response)
           
            serializer = LanguageModelSerializer(data_model)

            return Response(serializer.data)
        

