import sys
import openai
import llm_vm.onsite_llm as llms
from llm_vm.onsite_llm import load_model_closure
from llm_vm.agents.REBEL import agent
from llm_vm.completion.optimize import LocalOptimizer
import llm_vm.config as conf
from llm_vm.vector_db import PineconeDB
import os
import PyPDF2
from sentence_transformers import SentenceTransformer


if conf.settings.big_model is None:
    default_big_model = "chat_gpt"
else:
    default_big_model= conf.settings.big_model

if conf.settings.small_model is None:
    default_small_model= "pythia"
else:
    default_small_model = conf.settings.small_model

# Builds the client to facilitate easy use.
def client_build(type = "inference", openai_key = None, big_model = default_big_model, small_model =default_small_model,big_model_config={},small_model_config={}):
    if type == "inference":
        return Simple_Inference_Client(big_model, big_model_config, openai_key)
    else:
        return Client(big_model, small_model, big_model_config, small_model_config)
    
# Shoe-in for simple inference. Allows for traditional initialization of the client or dynamic initialization for rapid inference without agents.
class Simple_Inference_Client:
    def __init__(self, model = default_big_model, model_config = {}, openai_key = None):
        self.model = load_model_closure(model)(**model_config)
        openai.api_key = openai_key

    def complete(self, prompt, max_len=256, **kwargs):
        return self.model.generate(prompt, max_len,**kwargs)



class Client:
    """
    This is a class sets up a Local optimizer around a LLM for easy access.

    Attributes:
        optimizer (LocalOptimizer): Anarchy LLM Optimizer
        openai_key (str): API key for OpenAI access

    Methods:
        complete: The Anarchy completion endpoint giving access to a specified LLM
    """

    def __init__(self, big_model = default_big_model, small_model =default_small_model,big_model_config={},small_model_config={}):
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
        self.big_model = big_model
        self.small_model = small_model 
        self.teacher = load_model_closure(big_model)(**big_model_config)
        self.student = load_model_closure(small_model)(**small_model_config)
        self.vector_db = None
        self.emb_model = None

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

        # load the optimizer into object memory for use by the complete function
        self.optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None, call_big=self.CALL_BIG, call_small= CALL_SMALL,
                                        big_model = self.teacher, small_model = self.student)
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
                 openai_kwargs = {},
                 hf_kwargs = {}):
        """
        This function is Anarchy's completion entry point

        Parameters:
            prompt (str): Prompt to send to LLM for generation
            context (str): Unchanging context to send to the LLM for generation.  Defaults to "" and doesn't do fine-tuning.
            finetune (bool): Boolean value that begins fine tuning when set to True
            data_synthesis (bool): Boolean value to determine whether data should be synthesized for fine-tuning or not
            temperature (float): Sampling temperature for use during generation, between 0 and 2
            stoptoken (str | list): Sequence for stopping token generation
            tools (list): API tools for using with the Rebel agent
            openai_kwargs (dict): Key word arguments for generation with Open AI models
            hf_kwargs (dict): Key word arguments for generation with 


        Returns:
            str: LLM Generated Response

        Example:
           >>> SmallLocalOpt.generate("How long does it take for an apple to grow?)
           How long does it take for an apple tree to grow?
        """
        static_context = context
        dynamic_prompt = prompt
        use_rebel_agent = False

        if self.big_model in ['gpt', 'chat_gpt', 'gpt4']:
            kwargs = openai_kwargs
            if "temperature" not in kwargs:
                kwargs.update({"temperature":temperature})
            if stoptoken is not None:
                kwargs.update({"stop":stoptoken})
                
        else:
            kwargs = hf_kwargs
            if temperature > 0:
                kwargs.update({"do_sample": True})
                kwargs.update({"temperature":temperature})

        if openai_key is not None:
            self.openai_key = openai_key

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

        self.optimizer.openai_key = openai.api_key
        # self.agent.set_api_key(openai.api_key,"OPENAI_API_KEY") #
        if os.getenv("OPENAI_API_KEY") is None and use_rebel_agent==True :
            print("warning: you need OPENAI_API_KEY environment variable for ", file=sys.stderr)

        try:
            if not use_rebel_agent:
                completion = self.optimizer.complete(static_context,dynamic_prompt,data_synthesis=data_synthesis,finetune = finetune, **kwargs)
                print("COMPLETED RUNNING")
            else:
                completion = self.rebel_agent.run(static_context+dynamic_prompt,[])[0]
        except Exception as e:
            return {"status":0, "resp": str(e)}
        return {"completion":completion, "status": 200}

    def load_finetune(self, model_filename=None):
        self.teacher.load_finetune(model_filename)

    def set_pinecone_db(self, api_key, env_name):
        self.vector_db = PineconeDB(api_key, env_name)
        self.emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _pdf_loader(self, pdf_file_path):

        try:
        # Open the PDF file in read-binary mode
            with open(pdf_file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfFileReader(pdf_file)

                pdf_text = ''
                
                for page_num in range(pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)
                    pdf_text += page.extractText()
                
                pdf_file.close()
                
                return pdf_text
        
        except Exception as e:
            # Handle exceptions, such as file not found or invalid PDF format
            print(f"Error: {e}")
            return None
        
    def create_pinecone_index(self, **kwargs):
        self.vector_db.create_index(**kwargs)

    def store_pdf(self, pdf_file_path, chunk_size=1024, **kwargs):
        text = self._pdf_loader(pdf_file_path)
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = [self.emb_model.encode(chunk) for chunk in chunks]
        kwargs["vector"] = embeddings
        self.vector_db.upsert(**kwargs)

    def RAG_complete(self, prompt,
                 context = "",
                 openai_key = None,
                 finetune=False,
                 data_synthesis = False,
                 temperature=0,
                 stoptoken = None,
                 tools = None,
                 query_kwargs = {},
                 hf_kwargs = {},
                 openai_kwargs = {}):
        
        """
        This function is Anarchy's completion entry point

        Parameters:
            prompt (str): Prompt to send to LLM for generation
            context (str): Unchanging context to send to the LLM for generation.  Defaults to "" and doesn't do fine-tuning.
            finetune (bool): Boolean value that begins fine tuning when set to True
            data_synthesis (bool): Boolean value to determine whether data should be synthesized for fine-tuning or not
            temperature (float): Sampling temperature for use during generation, between 0 and 2
            stoptoken (str | list): Sequence for stopping token generation
            tools (list): API tools for using with the Rebel agent
            query_kwargs (dict): Key word arguments for querying the Pinecone vector database
            openai_kwargs (dict): Key word arguments for generation with Open AI models
            hf_kwargs (dict): Key word arguments for generation with 

        Returns:
            str: RAG Response
        """
        
        prompt_embeddings = self.emb_model.encode(prompt)
        query_kwargs["vector"] = prompt_embeddings
        if "top_k" not in query_kwargs:
            query_kwargs["top_k"] = 5

        topk_vecs = self.vector_db.query(**query_kwargs)

        decoded_vecs = self.emb_model.decode(topk_vecs)

        prompt = decoded_vecs + "/n" + prompt

        static_context = context
        dynamic_prompt = prompt
        use_rebel_agent = False

        if self.big_model in ['gpt', 'chat_gpt', 'gpt4']:
            kwargs = openai_kwargs
            if "temperature" not in kwargs:
                kwargs.update({"temperature":temperature})
            if stoptoken is not None:
                kwargs.update({"stop":stoptoken})       
        else:
            kwargs = hf_kwargs
            if temperature > 0:
                kwargs.update({"do_sample": True})
                kwargs.update({"temperature":temperature})

        if openai_key is not None:
            self.openai_key = openai_key

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

        self.optimizer.openai_key = openai.api_key
        # self.agent.set_api_key(openai.api_key,"OPENAI_API_KEY") #
        if os.getenv("OPENAI_API_KEY") is None and use_rebel_agent==True :
            print("warning: you need OPENAI_API_KEY environment variable for ", file=sys.stderr)

        try:
            if not use_rebel_agent:
                completion = self.optimizer.complete(static_context,dynamic_prompt,data_synthesis=data_synthesis,finetune = finetune, **kwargs)
            else:
                completion = self.rebel_agent.run(static_context+dynamic_prompt,[])[0]
        except Exception as e:
            return {"status":0, "resp": str(e)}
        return {"completion":completion, "status": 200}
