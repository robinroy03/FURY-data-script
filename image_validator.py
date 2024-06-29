"""
Takes an image, returns string describing it. Uses Ollama for local inference.
"""

import base64
import requests
import json


def image_description(path: str, prompt: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    encoded_data = base64.b64encode(data).decode('utf-8')

    obj = {
        "model": "moondream",
        "prompt": prompt,
        "images": [encoded_data],
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=obj)
    response_json = json.loads(response.text)
    
    return response_json['response']