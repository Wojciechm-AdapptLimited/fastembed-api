# FastEmbed API

## Overview

A simple REST API wrapper for FastEmbed library written using FastAPI. FastEmbed is a library that provides a simple interface to embed data in a fast and efficient way using Open Source models that can be run locally.

## Installation

It is recommended to create a virtual environment before installing the API. To create it run the following commands:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

Run the API with the following command (you can change the host and port as needed)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

API provides following endpoints:

### List available models

```bash
curl http://localhost:8000/api/models
```

### Get model details

```bash
curl http://localhost:8000/api/models/{model_name}
```

### Embed data
```bash
curl -X 'POST' \
  http://127.0.0.1:8000/api/embed \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": {model_name},
    "texts": [
        {text1},...
    ]
  }'
```

