from transformers import AutoTokenizer, BloomModel, BloomConfig
import torch

config = BloomConfig()
config.torchscript = True

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = BloomModel(config)
model.eval()
model = BloomModel.from_pretrained("bigscience/bloom-560m", torchscript=True)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
t1 = inputs['input_ids']
t2 = inputs['attention_mask']


module = torch.jit.trace(model, t1)
graph = module.graph.copy()

def parse(graph):
    for n in graph.nodes():
        #print(n.scopeName(), n.kind())
        attrs = str({k: k for k in n.attributeNames()})
        #print("attrs", attrs)
        inputs = [i for i in n.inputs()]
        #print("inputs", inputs)
        outputs = [i  for i in n.outputs()]
        #print("outputs", outputs)
        #print('---------------\n\n')

    return (attrs, inputs, outputs)
