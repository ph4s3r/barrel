"""Fast API RAG backend server."""
from fastapi import FastAPI, Response, Depends, status

from request_models import PromptArgs
from embedding_client_voyage import get_embedder_client
from vectordb_client import PineConeClient, get_pinecone_client, process_pc_qr
from llm_client import SuperPrompt


app = FastAPI(title="Barrel", docs_url="/")


@app.post("/user_prompt", response_model=str)
async def user_prompt(
    prompt: str,
    args: PromptArgs,
    client = Depends(get_embedder_client),
    pc_client: PineConeClient = Depends(get_pinecone_client),
):
    """Endpoint for processing user prompts."""
    ### embedding client client START
    result = client.embed(prompt, model="voyage-3-large", input_type="query")
    return_vector = result.embeddings[0]
    print(f"Embedding token length: {result.total_tokens}, vector dim: {len(return_vector)}")
    ### embedding client client END

    ### pinecone client START
    pinecone_response = pc_client.query(input_vector=return_vector, top_k=args.top_k)
    context_text = process_pc_qr(pinecone_response, mss=args.mss)
    vector_ids = [item.id for item in pinecone_response._data_store["matches"]]
    print("vector ids:", vector_ids)

    if context_text is None:
        scores = ", ".join(str(match.score) for match in pinecone_response.matches)
        return Response(
            status_code=status.HTTP_409_CONFLICT,
            content=f"No vectors with similarity score above the mss threshold: {args.mss}. MSS scores: [{scores}]"
        )
    ### pinecone client END

    ### LLM client START
    sp = SuperPrompt()
    outputs = sp.process_prompt(prompt, context_text)
    ### LLM client end

    return outputs[0]
