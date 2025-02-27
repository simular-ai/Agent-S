import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",  # Allows external access
        port=8000,       # Default port for FastAPI
        reload=True      # Auto-reload on code changes
    ) 