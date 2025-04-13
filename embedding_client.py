import requests


def send_embedding_request(prompt: str) -> dict:
    """Send question to 'embed' endpoint."""
    response = requests.post(
        "http://127.0.0.1:80/embed",
        json={"text": prompt},
        headers={"Content-Type": "application/json"},
        timeout=(1, 6)
    )
    response.raise_for_status()

    return response.json()
