# We are import the require library
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# We are define your custom dataset class if needed
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
# We are define the evaluation function
def evaluate_models(model_names, test_datset, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    for model_name in model_names:
        print(f"Evaluating model: {model_name}")
        
        # We are Load the fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(f"fine_tuned_{model_name}").to(device)
        
        # We are create data loader for evaluation
        test_loader = DataLoader(test_datset, batch_size=8, shuffle=False)
        
        # We are evaluation loop
        total_loss = 0.0
        total_samples = 0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evalatiing"):
                input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
                labels = input_ids.clone()
                
                outouts = model(input_ids, labels=labels)
                
                loss = outouts.loss
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)