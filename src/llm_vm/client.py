import openai
import llm_vm.onsite_llm as llms
from llm_vm.onsite_llm import load_model_closure
from llm_vm.agents.REBEL import agent
from llm_vm.completion.optimize import LocalOptimizer
import os




class Client:
    """
    This is a class sets up a Local optimizer around a LLM for easy access.

    Attributes:
        optimizer (LocalOptimizer): Anarchy LLM Optimizer
        openai_key (str): API key for OpenAI access

    Methods:
        complete: The Anarchy completion endpoint giving access to a specified LLM
    """

    def __init__(self, big_model = 'chat_gpt', small_model ='gpt',big_model_config={},small_model_config={}):
        """
        This __init__ function allows the user to specify which LLM they want to use upon instantiation.

        Parameters:
            big_model (str): Name of LLM to be used as a reliable source
            small_model (str): Name of small model to be used for fine-tuning

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
        print("Using model: " + big_model) # announce the primary LLM that is generating results

        # These functions allow for proper initialization of the optimizer
        # def CALL_BIG(prompt, max_len=256, **kwargs):

        #     return self.teacher.generate(prompt, max_len,**kwargs)

        def CALL_SMALL(prompt, max_len=256, **kwargs):

            return self.student.generate(prompt, max_len,**kwargs)

        # load the optimizer into object memory for use by the complete function
        self.optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None, call_big=self.CALL_BIG, call_small= CALL_SMALL,
                                        big_model = self.teacher, small_model = self.student)
        self.rebel_agent = agent.Agent("", [], verbose=1)

    # These functions allow for proper initialization of the optimizer
    def CALL_BIG(self, prompt, max_len=256, **kwargs):

        return self.teacher.generate(prompt, max_len,**kwargs)

    def complete(self, prompt,
                 context,
                 openai_key = "",
                 finetune=False,
                 data_synthesis = False,
                 temperature=0,
                 stoptoken = None,
                 tools = None):
        """
        This function is Anarchy's completion entry point

        Parameters:
            prompt (str): Prompt to send to LLM for generation
            context (str): Context to send to the LLM for generation
            finetune (bool): Boolean value that begins fine tuning when set to True
            data_synthesis (bool): Boolean value to determine whether data should be synethesized for fine-tuning or not
            temperature (float): An analog


        Returns:
            str: LLM Generated Response

        Example:
           >>> Small_Local_OPT.generate("How long does it take for an apple to grow?)
           How long does it take for an apple tree to grow?
        """
        static_context = context
        dynamic_prompt = prompt
        use_rebel_agent = False
        kwargs = {}

        kwargs.update({"temperature":temperature})

        if openai_key:
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
                self.rebel_agent.set_tools(final_tools)
                use_rebel_agent = True

        try:
            if openai_key:
                openai.api_key = self.openai_key
        except:
            return  {"status":0, "resp":"Issue with OpenAI key"}

        self.optimizer.openai_key = openai.api_key
        agent.set_api_key(openai.api_key,"OPENAI_API_KEY")

        try:
            if not use_rebel_agent:
                completion = self.optimizer.complete(static_context,dynamic_prompt,data_synthesis=data_synthesis,finetune = finetune, **kwargs)
            else:
                completion = self.rebel_agent.run(static_context+dynamic_prompt,[])[0]
        except Exception as e:
            return {"status":0, "resp": str(e)}
        return {"completion":completion, "status": 200}

    def load_finetune(self, model_filename=None):
        self.teacher.load_finetune(model_filename)

