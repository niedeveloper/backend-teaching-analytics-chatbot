from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v2 import unified_chat # unified chat
from app.core.config import settings

app = FastAPI(title="NIE Backend API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://niefrontend.reallycoolworkspace.uk"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
#app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
#app.include_router(langchainchat.router, prefix="/api/v1", tags=["langchain"])
#app.include_router(general_chat.router, prefix="/api/v1", tags=["general_chat"])  # Add basic chatbot
app.include_router(unified_chat.router, prefix="/api/v2", tags=["unified_chat"])  # Add unified chat

@app.get("/")
async def root():
    return {"message": "NIE Backend API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
