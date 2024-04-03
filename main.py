import numpy as np
import os

from typing import Any, Iterable, Annotated
from fastembed.embedding import TextEmbedding as Embedding
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache/")


class ModelRequest(BaseModel):
    """
    Represents a request object for info about embedding model.

    Attributes:
        model (str): The name of the model.
    """

    model: str


class EmbeddingRequest(ModelRequest):
    """
    Represents a request object for text embedding.

    Attributes:
        passages (list[str]): The list of passages to process.
    """

    texts: list[str]


def get_model_info(model: str) -> dict[str, Any]:
    models = Embedding.list_supported_models()
    models = list(filter(lambda x: x["model"] == model, models))

    if not models:
        raise ValueError(f"Model {model} doesn't exist.")

    model_info = models[0]

    if "hf" in model_info["sources"]:
        path_name = model_info["sources"]["hf"]
        path_name = path_name.split("/")
        path_name = f"models--{path_name[0]}--{path_name[1]}"
    elif "url" in model_info["sources"]:
        path_name = model_info["sources"]["url"]
        path_name = path_name.split("/")[-1]
        path_name = path_name.split(".")[0]
    else:
        raise ValueError(f"Model {model} doesn't have a valid source.")

    path_name = os.path.join(cache_dir, path_name)

    model_info["cached"] = os.path.exists(path_name)

    return model_info


@app.get("/")
async def root() -> str:
    return "Hello World!"


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "OK"}


@app.get("/api/models")
async def get_models() -> list[str]:
    """
    Retrieves a list of supported model names for embedding.

    Returns:
        A list of model names.
    """
    models = Embedding.list_supported_models()
    model_names = [model["model"] for model in models]
    return model_names


@app.get("/api/models/{model:path}")
async def get_model(
    model: Annotated[str, Path(title="The name of the model to GET")]
) -> dict[str, Any]:
    """
    Retrieves information about a specific model from the provider.

    Args:
        model (str): The name of the model

    Returns:
        dict[str, Any]: A dictionary containing information about the model.

    Raises:
        HTTPException: If the model is not found.
    """
    try:
        model_info = get_model_info(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return model_info


@app.post("/api/embed", response_model=dict[str, list[list[float]]])
async def embed(request: EmbeddingRequest) -> dict[str, Iterable[np.ndarray]]:
    """
    Embeds the given passages using the specified model.

    Args:
        request (EmbeddingRequest): The request object containing the model and texts to be embeded.

    Returns:
        dict[str, Iterable[np.ndarray]]: A dictionary containing the embedded passages.

    Raises:
        HTTPException: If the model doesn't exist, wasn't pulled, or if there was an error embedding the passages.
    """
    try:
        embedding = Embedding(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        embeddings = list(embedding.embed(request.texts))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"embedding": embeddings}
