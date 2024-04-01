import requests
import json
# URL of the JSON file
url = 'https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json'
# Send HTTP request to the URL
response = requests.get(url)
# Load the JSON data from the response
data = json.loads(response.text)
# Specify the local path where you want to store the JSON file
file_path = 'local_vocab.json'
# Write the JSON data to a file in the specified path
with open(file_path, 'w') as file:
    json.dump(data, file)
