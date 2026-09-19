import os
import sys
from pathlib import Path

# Fix OpenMP duplicate runtime error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure parent and current directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_TITLE, API_VERSION, SAMPLES_DIR, API_HOST, API_PORT
from backend.data.sample_generator import generate_sample_tiles
from backend.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("satquery.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SatQuery AI Backend...")
    # Pre-generate sample tiles if absent
    if not SAMPLES_DIR.exists() or len(list(SAMPLES_DIR.glob("*.npy"))) == 0:
        logger.info("Initializing sample multispectral satellite tiles...")
        generate_sample_tiles(SAMPLES_DIR)
    yield
    logger.info("Shutting down SatQuery AI Backend...")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Earth Observation Multimodal VLM & AI Agent Framework for Multispectral Sentinel-2 Querying",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Ensure Swagger UI recognizes array of UploadFile as file inputs
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        if "properties" in schema:
            for prop in schema["properties"].values():
                if prop.get("type") == "array" and "items" in prop:
                    if prop["items"].get("contentMediaType") == "application/octet-stream":
                        prop["items"]["format"] = "binary"
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
