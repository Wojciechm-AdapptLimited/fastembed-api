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


class PassagesRequest(BaseModel):
    """
    Represents a request object for passages embedding.

    Attributes:
        provider (str): The provider of the model to use for processing the passages.
        model (str): The model to use for processing the passages.
        passages (list[str]): The list of passages to process.
    """

    provider: str
    model: str
    passages: list[str]


class QueryRequest(BaseModel):
    """
    Represents a request object for query embedding.

    Attributes:
        provider (str): The provider of the model to use for processing the query.
        model (str): The model to use for processing the query.
        query (str): The query to process.
    """

    provider: str
    model: str
    query: str


def get_model_name(provider: str, model: str) -> str:
    return f"{provider}/{model}"


async def get_model_path(provider: str, model: str) -> str:
    model_name = get_model_name(provider, model)
    model_info = await get_model_info(provider, model)

    if "hf" in model_info["sources"]:
        path_name = model_info["sources"]["hf"]
        path_name = path_name.split("/")
        path_name = f"models--{path_name[0]}--{path_name[1]}"
    elif "url" in model_info["sources"]:
        path_name = model_info["sources"]["url"]
        path_name = path_name.split("/")[-1]
        path_name = path_name.split(".")[0]
    else:
        raise ValueError(f"Model {model_name} doesn't exist.")
    
    path_name = os.path.join(os.getcwd(), cache_dir, path_name)
        
    if not os.path.exists(path_name):
        raise ValueError(f"Model {model_name} wasn't pulled.")

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


@app.get("/api/models/{provider}/{model}")
async def get_model_info(provider: str, model: str) -> dict[str, Any]:
    """
    Retrieves information about a specific model from the provider.

    Args:
        provider (str): The provider of the model.
        model (str): The name of the model.

    Returns:
        dict[str, Any]: A dictionary containing information about the model.

    Raises:
        HTTPException: If the model is not found.
    """
    
    model_name = f"{provider}/{model}"
    models = Embedding.list_supported_models()
    models = list(filter(lambda x: x["model"] == model_name, models))

    if not models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")

    model_info = models[0]

    return model_info


@app.post("/api/pull/{provider}/{model}")
async def pull_model(provider: str, model: str) -> dict[str, str]:
    """
    Pulls a model from the specified provider.

    Args:
        provider (str): The provider of the model.
        model (str): The name of the model.

    Returns:
        dict[str, str]: A dictionary containing a success message.
    
    Raises:
        HTTPException: If the model could not be pulled.
    """
    model_name = f"{provider}/{model}"
    try:
        _ = Embedding(model_name, max_length=512)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"Model {model_name} pulled successfully."}


@app.delete("/api/delete/{provider}/{model}")
async def delete_model(provider: str, model: str) -> dict[str, str]:
    """
    Deletes a model from the specified provider.

    Args:
        provider (str): The provider of the model.
        model (str): The name of the model to delete.

    Returns:
        dict[str, str]: A dictionary containing a success message.
    
    Raises:
        HTTPException: If the model could not be deleted (e.g. if it doesn't exist or wasn't pulled)
    """
    model_name = f"{provider}/{model}"

    try:
        path_name = await get_model_path(provider, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    shutil.rmtree(path_name)

    return {"message": f"Model {model_name} deleted successfully."}


@app.post("/api/embed_passages", response_model=dict[str, Iterable[list[float]]])
async def embed_passages(request: PassagesRequest) -> dict[str, Iterable[np.ndarray]]:
    """
    Embeds the given passages using the specified model.

    Args:
        request (PassagesRequest): The request object containing the model and passages.

    Returns:
        dict[str, Iterable[np.ndarray]]: A dictionary containing the embedded passages.
    
    Raises:
        HTTPException: If the model doesn't exist, wasn't pulled, or if there was an error embedding the passages.
    """
    provider = request.provider
    model = request.model
    passages = request.passages
        
    try:
        _ = await get_model_path(provider, model)
        embedding = Embedding(get_model_name(provider, model), max_length=512)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        embeddings = embedding.passage_embed(passages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {"embedding": embeddings}


@app.post("/api/embed_query", response_model=dict[str, list[list[float]]])
async def embed_query(request: QueryRequest) -> dict[str, Iterable[np.ndarray]]:
    """
    Embeds a query using a specified model.

    Args:
        request (QueryRequest): The query request object containing the model and query.

    Returns:
        dict[str, Iterable[np.ndarray]]: A dictionary containing the embedded query.

    Raises:it 
        HTTPException: If there is an error in the request or embedding process.
    """
    provider = request.provider
    model = request.model
    query = request.query
    
    try:
        _ = await get_model_path(provider, model)
        embedding = Embedding(get_model_name(provider, model), max_length=512)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        embeddings = embedding.query_embed(query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {"embedding": embeddings}
