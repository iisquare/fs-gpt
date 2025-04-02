import tiktoken
from typing import Dict

from fastapi import APIRouter, Depends, status

from fs_gpt.core.Server import Server
from fs_gpt.embedding.RAGEmbedding import RAGEmbedding
from fs_gpt.protocol.embedding import EmbeddingCreateParams


def create_router(app: Server, rag: RAGEmbedding) -> APIRouter:

    router = APIRouter()

    @router.post(
        "/embeddings",
        dependencies=[Depends(app.check_api_key)],
        status_code=status.HTTP_200_OK,
    )
    @router.post(
        "/engines/{model_name}/embeddings",
    )
    async def create_embeddings(request: EmbeddingCreateParams, model_name: str = None,):
        """Creates embeddings for the text"""
        if request.model is None:
            request.model = model_name

        request.input = request.input
        if isinstance(request.input, str):
            request.input = [request.input]
        elif isinstance(request.input, list):
            if isinstance(request.input[0], int):
                decoding = tiktoken.model.encoding_for_model(request.model)
                request.input = [decoding.decode(request.input)]
            elif isinstance(request.input[0], list):
                decoding = tiktoken.model.encoding_for_model(request.model)
                request.input = [decoding.decode(text) for text in request.input]

        request.dimensions = request.dimensions or app.arguments().get("embedding_size", -1)

        return rag.encode(
            texts=request.input,
            model=request.model,
            encoding_format=request.encoding_format,
            dimensions=request.dimensions,
        )
    return router


def main(args: Dict):
    app = Server(args)
    rag = RAGEmbedding(args)
    app.include_router(create_router(app, rag), tags=["Embedding"])
    app.run()
