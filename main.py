from fastapi import FastAPI, Response

from embedding_client import send_embedding_request
from vectordb_client import PineConeClient, process_pc_qr
from llm_client import SuperPrompt
from request_models import Mss


app = FastAPI(
    title="Barrel",
    docs_url="/"
)

pc_client = PineConeClient()


@app.post("/user_prompt", response_model=str)
async def user_prompt(prompt: str, mss: Mss):
    """Endpoint for processing user prompts."""


    ### embedding client client START
    return_vector = send_embedding_request(prompt)
    print("Response Vector dim:", len(return_vector[0]))
    ### embedding client client END


    ### pinecone client START
    pinecone_response = pc_client.query(inputvector=return_vector,top_k=3)
    mss = 0.3
    context_text = process_pc_qr(pinecone_response,mss=mss)
    if context_text is None:
        return Response(status_code=409, content=f"No vectors with similiarity score above the mss {mss} found")
    ### pinecone client END



    ### LLM client START
    sp = SuperPrompt()
    outputs = sp.process_prompt(prompt, context_text)
    ### LLM client end

    # breakpoint()

    return outputs[0]
