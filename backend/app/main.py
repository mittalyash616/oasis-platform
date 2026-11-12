from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.api.v1.endpoints import router as api_v1_router

app = FastAPI(
    title="OASIS — Oil-spill Attribution & Satellite Intelligence System",
    description="Marine Forensics Platform for the Indian Coast Guard. Automated SAR detection, hindcast drift modeling, and AIS vessel attribution API. Team Alpha.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root traffic directly to Swagger documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "pipeline_ready": True}
