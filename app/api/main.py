"""FastAPI application main file."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio Risk Intelligence API",
    description="AI-powered explainable portfolio risk analysis system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Full-stack: serve frontend static files
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        """Serve frontend SPA (data extraction UI)."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"service": "Portfolio Risk Intelligence API", "version": "1.0.0", "status": "operational"}
else:
    @app.get("/")
    async def root():
        """Root endpoint when static folder is not present."""
        return {
            "service": "Portfolio Risk Intelligence API",
            "version": "1.0.0",
            "status": "operational"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
