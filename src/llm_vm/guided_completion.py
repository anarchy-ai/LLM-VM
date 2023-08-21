import outlines.models as models 
import outlines.text.generate as generate

model = models.transformers("gpt2-medium")

#this class is called when optimize.complete is called with the regex parameter
class RegexCompletion:
    def complete(prompt,regex):
        guided = generate.regex(model,regex)(prompt)
        return guided

#this class is called when optimize.complete is called with the choices parameter
class ChoicesCompletion:
    def complete(prompt,choices):
        guided = generate.choice(model,choices)(prompt)
        return guided
    
#this class is called with optimize.complete is caled with the type parameter
class TypeCompletion:
    def complete(prompt,type):
        if type != "float" and type != "integer":
            raise Exception("type must be float or integer")
        guided = getattr(generate,type)(model)(prompt)
        return guided