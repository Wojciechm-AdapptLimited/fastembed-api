import numpy as np
import os
import shutil

from typing import Any, Iterable
from fastembed.embedding import TextEmbedding as Embedding
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "cache")


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


async def get_model_path(request: ModelRequest) -> str:
    model_info = await get_model_info(request)

    if "hf" in model_info["sources"]:
        path_name = model_info["sources"]["hf"]
        path_name = path_name.split("/")
        path_name = f"models--{path_name[0]}--{path_name[1]}"
    elif "url" in model_info["sources"]:
        path_name = model_info["sources"]["url"]
        path_name = path_name.split("/")[-1]
        path_name = path_name.split(".")[0]
    else:
        raise ValueError(f"Model {request.model} doesn't exist.")

    path_name = os.path.join(os.getcwd(), cache_dir, path_name)

    if not os.path.exists(path_name):
        raise ValueError(f"Model {request.model} wasn't pulled.")

    return path_name


@app.get("/")
async def root() -> str:
    return "Hello World!"


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


@app.post("/api/info")
async def get_model_info(request: ModelRequest) -> dict[str, Any]:
    """
    Retrieves information about a specific model from the provider.

    Args:
        request (ModelRequest): The request object containing the model name.

    Returns:
        dict[str, Any]: A dictionary containing information about the model.

    Raises:
        HTTPException: If the model is not found.
    """

    models = Embedding.list_supported_models()
    models = list(filter(lambda x: x["model"] == request.model, models))

    if not models:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found.")

    model_info = models[0]

    return model_info


@app.post("/api/pull")
async def pull_model(request: ModelRequest) -> dict[str, str]:
    """
    Pulls a model from the specified provider.

    Args:
        request (ModelRequest): The request object containing the model name.

    Returns:
        dict[str, str]: A dictionary containing a success message.

    Raises:
        HTTPException: If the model could not be pulled.
    """
    try:
        _ = Embedding(request.model, max_length=512)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"Model {request.model} pulled successfully."}


@app.delete("/api/delete")
async def delete_model(request: ModelRequest) -> dict[str, str]:
    """
    Deletes a model from the specified provider.

    Args:
        request (ModelRequest): The request object containing the model name.

    Returns:
        dict[str, str]: A dictionary containing a success message.

    Raises:
        HTTPException: If the model could not be deleted (e.g. if it doesn't exist or wasn't pulled)
    """
    try:
        path_name = await get_model_path(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    shutil.rmtree(path_name)

    return {"message": f"Model {request.model} deleted successfully."}


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
        _ = await get_model_path(request)
        embedding = Embedding(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        embeddings = list(embedding.embed(request.texts))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"embedding": embeddings}
