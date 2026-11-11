from fastapi import FastAPI

app = FastAPI(
    title="ClauseGuard API",
    description="AI-powered legal document analysis",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "ClauseGuard API is running"}