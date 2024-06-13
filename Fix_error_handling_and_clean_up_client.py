# We are importing the require library
import os
import requests
import json

def generate_text(prompt, max_length=50):
    try:
        # Retrieve OpenAI API key from environment variable
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise EnvironmentError("OpenAI API key not found in environmnet variables")
        
        url = "https://api/openai.com/v1/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer{openai_key}"
        }
        
        # We are prepare JSON payload
        data = {
            "model": "text-davinci-003",
            "prompt": prompt,
            "max_tokens": max_length
        }
        
        # We are making a POST request to OpenAI API
        response = requests.post(url, headers=headers)
        # Raise HTTPError for non-2xx status codes
        response.raise_for_status()
        
        # We are parse response JSON
        result = response.json()
        generate_text = result['choices'][0]['text'].strip()
        return generate_text
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        

# We are define the entry poing of the function
if __name__ == "__main__":
    prompt = "What is the meaning of life?"
    max_length = 100
    
    generate_text = generate_text(prompt, max_length)
    if generate_text:
        print("Generated text:", generate_text)
    else:
        print("Failed to generate text")
