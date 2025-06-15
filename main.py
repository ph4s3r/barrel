"""Fast API RAG backend server."""
from fastapi import FastAPI, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from voyageai.client import Client

from request_models import PromptArgs
from embedding_client_voyage import get_embedder_client
from vectordb_client import PineConeClient, get_pinecone_client
from llm_client_azure import SuperPrompt


app = FastAPI(title="Barrel", docs_url="/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # or ["http://localhost:3000"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/user_prompt", response_model=str)
async def user_prompt(
    prompt: str,
    args: PromptArgs,
    client: Client = Depends(get_embedder_client),
    pc_client: PineConeClient = Depends(get_pinecone_client)
):
    """Endpoint for processing user prompts."""
    ### embedding client client START
    print("[PROCESSING USER QUERY]", "*"*90)
    print(f"[PROMPT]: {prompt}")
    result = client.embed(prompt, model="voyage-3-large", input_type="query")
    return_vector = result.embeddings[0]
    print(f"[EMBEDDING TOKEN LEN]: {result.total_tokens}")
    ### embedding client client END

    ### pinecone client START
    pinecone_response = pc_client.query(input_vector=return_vector, top_k=args.top_k)
    context_text = pinecone_response.matches
    vector_ids = [item.id for item in pinecone_response.matches]
    print(f"[VECTOR IDS]: {vector_ids}")

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

    return outputs

@app.get("/indexes")
async def get_indexes(pc_client: PineConeClient = Depends(get_pinecone_client)):
    """Endpoint for retrieving sources from the cached vector data."""
    sources = pc_client.return_sources()
    
    if sources is None:
        return Response(
            status_code=status.HTTP_404_NOT_FOUND,
            content="No cached vectors found"
        )
    
    # Transform sources into a more JSON-friendly format
    sources_dict = {source: count for source, count in sources}
    
    return {"sources": sources_dict}
