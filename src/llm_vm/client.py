import sys
import openai
from llm_vm.onsite_llm import load_model_closure
from llm_vm.agents.REBEL import agent
from llm_vm.completion.optimize import LocalOptimizer
import llm_vm.config as conf
import os

from llm_vm.utils.speculative_sampling import speculative_sampling, speculative_sampling_v2

if conf.settings.big_model is None:
    default_big_model = "chat_gpt"
else:
    default_big_model= conf.settings.big_model

if conf.settings.small_model is not  None:
    default_small_model= "pythia"
else:
    default_small_model = conf.settings.small_model




class Client:
    """
    This is a class sets up a Local optimizer around a LLM for easy access.

    Attributes:
        optimizer (LocalOptimizer): Anarchy LLM Optimizer
        openai_key (str): API key for OpenAI access

    Methods:
        complete: The Anarchy completion endpoint giving access to a specified LLM
    """

    def __init__(self, big_model = default_big_model,
                 small_model =default_small_model,big_model_config={},small_model_config={},
                 optimizer="local_optimizer", speculative_sampling_tokenizer="small", speculative_sampling_version="google"):
        """
        This __init__ function allows the user to specify which LLM they want to use upon instantiation.

        Parameters:
            big_model (str): Name of LLM to be used as a reliable source
            small_model (str): Name of small model to be used for fine-tuning
            optimizer (str): Name of optimizer ("local_optimizer" or "speculative_sampling")
            speculative_sampling_tokenizer (str): Name of tokenizer to be used for speculative sampling because both
                                                models should use the same tokenizer!  ("small" or "big")
            speculative_sampling_version (str): Name of speculative sampling version to be used ("google" or "deepmind")

        Returns:
            None

        Example:
            >>> client = Client(big_model = 'neo')
        """
        self.teacher = load_model_closure(big_model)(**big_model_config)
        self.student = load_model_closure(small_model)(**small_model_config)

        ## FIXME, do something like $HOME/.llm_vm/finetuned_models/
        if os.path.isdir("finetuned_models") == False:
            os.mkdir("finetuned_models")
        # Specify the model strategy the user will use
        # MODELCONFIG = models.ModelConfig(big_model=big_model, small_model=small_model)
        print("Using model: " + big_model, file=sys.stderr) # announce the primary LLM that is generating results

        # These functions allow for proper initialization of the optimizer
        # def CALL_BIG(prompt, max_len=256, **kwargs):

        #     return self.teacher.generate(prompt, max_len,**kwargs)

        def CALL_SMALL(prompt, max_len=256, **kwargs):
            small_model_filename = kwargs.get('small_model_filename', None)
            if small_model_filename is not None:
                self.student.load_finetune(small_model_filename)

            return self.student.generate(prompt, max_len,**kwargs)
        if optimizer == "local_optimizer":
            # load the optimizer into object memory for use by the complete function
            self.optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None, call_big=self.CALL_BIG, call_small= CALL_SMALL,
                                            big_model = self.teacher, small_model = self.student)
        elif optimizer == "speculative_sampling":
            self.optimizer = None
            self.speculative_sampling_version = speculative_sampling_version
            if speculative_sampling_tokenizer == 'small':
                self.speculative_sampling_tokenizer = self.student.tokenizer
            elif speculative_sampling_tokenizer == 'big':
                self.speculative_sampling_tokenizer = self.teacher.tokenizer
            else:
                raise Exception("speculative_sampling_tokenizer must be either 'small' or 'big'")
        else:
            raise Exception("optimizer must be either 'local_optimizer' or 'speculative_sampling'")

        self.rebel_agent = None # only initialized on first use

    # These functions allow for proper initialization of the optimizer
    def CALL_BIG(self, prompt, max_len=256, **kwargs):

        return self.teacher.generate(prompt, max_len,**kwargs)

    def complete(self, prompt,
                 context = "", # TODO: change back to static_context because otherwise 
                 openai_key = None,
                 finetune=False,
                 data_synthesis = False,
                 temperature=0,
                 stoptoken = None,
                 tools = None,
                 **kwargs):
        """
        This function is Anarchy's completion entry point

        Parameters:
            prompt (str): Prompt to send to LLM for generation
            context (str): Unchanging context to send to the LLM for generation.  Defaults to "" and doesn't do fine-tuning.
            finetune (bool): Boolean value that begins fine tuning when set to True
            data_synthesis (bool): Boolean value to determine whether data should be synthesized for fine-tuning or not
            temperature (float): An analog


        Returns:
            str: LLM Generated Response

        Example:
           >>> SmallLocalOpt.generate("How long does it take for an apple to grow?)
           How long does it take for an apple tree to grow?
        """
        static_context = context
        dynamic_prompt = prompt
        use_rebel_agent = False
        # TODO(us): speculative_sampling gives error when temp is 0,
        # self.optimizer is None means `speculative_sampling` is used
        if temperature == 0 and self.optimizer is None:
            temperature = 0.2
        else:
            raise Exception("temperature can't only be 0 for speculative sampling")
        kwargs.update({"temperature":temperature})

        if openai_key is not None:
            self.openai_key = openai_key

        if stoptoken is not None:
            kwargs.update({"stop":stoptoken})

        if tools is not None:
            if type(tools) != list:
                return Exception("Wrong data type for tools. Should be a list")
            else:
                final_tools=[]
                for i in tools:
                    temp_tool_dict = {}
                    temp_args_dict = {}
                    temp_tool_dict.update({"description":i["description"]})
                    temp_tool_dict.update({"dynamic_params":i["dynamic_params"]})
                    temp_tool_dict.update({"method":i["method"]})
                    temp_args_dict.update({"url":i["url"]})
                    temp_args_dict.update({"params":{}})
                    for j in i["static_params"].keys():
                        temp_args_dict["params"].update({j:i["static_params"][j]})
                    for k in i["dynamic_params"].keys():
                        temp_args_dict["params"].update({k:"{"+k+"}"})
                    temp_tool_dict.update({"args":temp_args_dict})
                    final_tools.append(temp_tool_dict)
                if self.rebel_agent is None:
                    self.rebel_agent = agent.Agent("", [], verbose=1) # initialize only if tools registered
                self.rebel_agent.set_tools(final_tools)
                use_rebel_agent = True

        try:
            if openai_key is not None:
                openai.api_key = self.openai_key
        except:
            return  {"status":0, "resp":"Issue with OpenAI key"}

        if self.optimizer is not None:  # speculative sampling works only with onsite models!
            self.optimizer.openai_key = openai.api_key
        # self.agent.set_api_key(openai.api_key,"OPENAI_API_KEY") # 

        if os.getenv("OPENAI_API_KEY") is None and use_rebel_agent==True :
            print("warning: you need OPENAI_API_KEY environment variable for ", file=sys.stderr)

        try:
            if not use_rebel_agent and isinstance(self.optimizer, LocalOptimizer):
                completion = self.optimizer.complete(static_context,dynamic_prompt,data_synthesis=data_synthesis,finetune = finetune, **kwargs)
            elif not use_rebel_agent and self.optimizer is None:
                input_ids = self.speculative_sampling_tokenizer.encode(static_context+dynamic_prompt, return_tensors='pt')
                if self.speculative_sampling_version == "google":
                    output = speculative_sampling(input_ids, self.student.model, self.teacher.model, 256, temperature=temperature,
                                                  top_k=10, top_p=0.9)  # temperature=0 gives this error `probability tensor contains either `inf`, `nan` or element < 0`
                elif self.speculative_sampling_version == "deepmind":
                    output = speculative_sampling_v2(input_ids, self.student.model, self.teacher.model,256, temperature=temperature)

                completion = self.speculative_sampling_tokenizer.decode(output[0], skip_special_tokens=True)
            else:
                completion = self.rebel_agent.run(static_context+dynamic_prompt, [])[0]
        except Exception as e:
            return {"status":0, "resp": str(e)}
        return {"completion":completion, "status": 200}

    def load_finetune(self, model_filename=None):
        self.teacher.load_finetune(model_filename)
