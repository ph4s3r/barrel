import requests


def send_embedding_request(prompt: str): 

    response = requests.post(
        "http://127.0.0.1:80/embed", 
        json={"text": prompt}, 
        headers={"Content-Type": "application/json"}
    )

    response.raise_for_status()

    return response.json()
