import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# We are define an function to preprocess data
def preprocess_data(csv_file, tokenizer_name, max_length=512, test_size=0.1, random_state=42):
    # We first load the csv file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # We are Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # We are tokenize text and add tokenized sequences to DataFrame
    tokenized_sequences = []
    # adjust text_columns according to your dataset
    for text in df['text_columns']:
        tokenized_text = tokenizer.encode(text, max_length=max_length, truncation=True)
        tokenized_sequences.append(tokenized_text)
        
    df['tokenzied_text'] = tokenized_sequences
    
    # We are split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return train_df, test_df

# We are define the entrypoint of the function
if __name__ == "__main__":
    # We are setting the parameters
    csv_file = "path/to/your/dataset.csv"
    tokenizer_name = "tokenizer-name"
    max_length = 512
    test_size = 0.1
    random_state = 42
    
    # We preprocess data
    train_df, test_df = preprocess_data(csv_file, tokenizer_name, max_length=max_length, test_size=test_size)
    
    # We are saving the preprocessed data if needed
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    
    # waiting for future improvement  
