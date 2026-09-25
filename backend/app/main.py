from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.exports import router as exports_router
from app.api.nlp import router as nlp_router
from app.api.query import router as query_router
from app.api.retrieval import router as retrieval_router
from app.api.semantic import router as semantic_router
from app.api.videos import router as videos_router
from app.core.config import settings
from app.db.migrations import run_startup_migrations
from app.db.session import Base, engine


def create_app() -> FastAPI:
    settings.ensure_directories()
    run_startup_migrations(engine)
    Base.metadata.create_all(bind=engine)

    app = FastAPI(
        title="AI Traffic Video Assistant",
        version="0.1.0",
        description="Computer Vision + Vietnamese NLP video retrieval assistant for traffic surveillance.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/media", StaticFiles(directory=settings.upload_dir), name="media")
    app.include_router(videos_router, prefix="/api/videos", tags=["videos"])
    app.include_router(query_router, prefix="/api/query", tags=["query"])
    app.include_router(nlp_router, prefix="/api/nlp", tags=["nlp"])
    app.include_router(retrieval_router, prefix="/api/retrieval", tags=["retrieval"])
    app.include_router(semantic_router, prefix="/api/semantic", tags=["semantic"])
    app.include_router(exports_router, prefix="/api/exports", tags=["exports"])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
