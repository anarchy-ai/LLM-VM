import outlines.models as models
import outlines.text.generate as generate

model = models.transformers("gpt2")


class Completion:
    """
    A class used to generate completions when optimize.complete is called
    """

    def __init__(self, generator, *generator_args):
        """
        :param generator Callable[[Transformers, ...generator_args], None]:
            Generator function to be used on the complete
        :param *generator_args [Any]: Generator arguments (without model)
        """
        self.generator = generator
        self.generator_args = generator_args

    def complete(self, prompt):
        return self.generator(model, *self.generator_args)(prompt)

    @staticmethod
    def regex_completion(regex):
        return Completion(generate.regex, regex)

    @staticmethod
    def choices_completion(choices):
        return Completion(generate.choice, choices)

    @staticmethod
    def type_completion(type_name):
        if type_name not in ["float", "integer"]:
            raise Exception("type must be float or integer")
        return Completion(getattr(generate, type_name))

    @staticmethod
    def response_completion():
        return Completion(lambda _: (lambda x: x['response']))

    @staticmethod
    def create(regex, type, choices, default=None):
        completion = default
        if regex is not None:
            completion = Completion.regex_completion(regex)
        elif type is not None:
            completion = Completion.type_completion(type)
        elif choices is not None:
            completion = Completion.choices_completion(choices)
        return completion
