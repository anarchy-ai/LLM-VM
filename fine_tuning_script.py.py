# We are importing the library
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# We are define the custom dataset class if needed
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
# We are define the fine-tuning function
def fine_tune_models(model_names, train_dataset, tokenizer, num_epochs=3, batch_size=8,device="cuda" if torch.cuda.is_available() else "cpu"):
    for model_name in model_names:
        print(f"Fine-tuning model: {model_name}")
        
        # We are load the pre-trained model
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # We are define optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Creare Data Loader for training
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # We are Fine-tuning loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
                labels = input_ids.clone()
                
                optimizer.zero_grad()
                outputs = model(input_ids, labels=labels)
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            # We are print the average loss for the epoch
            print(f"Average Loss: {total_loss/ len(train_loader)}")
        
        # We are save the fine-tuned model
        model.save_pretrained(f"fine_tuned_{model_name}")
        
if __name__ == "__main__":
    # we are define the models to fine-tune
    model_names = [
        "facebook/opt-125m",
        "facebook/opt-350m",
    ]
    
    # We are Load preprocessed data and tokenizer
    train_dataset = CustomDataset(...)
    tokenizer = AutoTokenizer.from_pretrained("tokenizer-name")
    
    # We are fine-tune the models
    fine_tune_models(model_names, train_dataset, tokenizer)