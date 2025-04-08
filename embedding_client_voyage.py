import voyageai


def send_embedding_request(prompt: str): 

    vo = voyageai.Client()
   
    result = vo.embed(
        prompt, 
        model="voyage-3-large", 
        input_type="query")

    print(f"received the query embedding with the total number of {result.total_tokens} tokens")

    return result.embeddings[0]
