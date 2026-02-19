"""Main FastAPI application for CountyIQ API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import demographics, legal, property, search, upload

# Configure loguru
logger.add("logs/api.log", rotation="1 day", retention="7 days")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    logger.info("Starting CountyIQ API...")
    yield
    logger.info("Shutting down CountyIQ API...")


app = FastAPI(
    title="CountyIQ API",
    description="API for county-level property, legal, and demographics intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(property.router, prefix="/api/v1/property", tags=["property"])
app.include_router(legal.router, prefix="/api/v1/legal", tags=["legal"])
app.include_router(demographics.router, prefix="/api/v1/demographics", tags=["demographics"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": "CountyIQ API",
        "version": "1.0.0",
        "status": "healthy",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
