# main.py
from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    app = FastAPI(title="AI Crypto Trading API", version=os.getenv("APP_VERSION", "0.1.0"))

    # CORS (adjust as needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"message": "API is running"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Example lazy-import endpoint so module import never fails
    @app.get("/version")
    async def version():
        try:
            # Put heavy/optional imports here so they don't break uvicorn import
            import torch  # noqa: F401
        except Exception as e:
            # If something is missing in your environment, you’ll see it only when hitting this route
            raise HTTPException(status_code=500, detail=f"Optional import failed: {e}")
        return {"app_version": app.version}

    return app

# Uvicorn looks for this symbol: "main:app"
app = create_app()

if __name__ == "__main__":
    # Allows: python main.py
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
