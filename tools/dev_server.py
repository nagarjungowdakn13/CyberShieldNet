from fastapi import FastAPI

app = FastAPI(title="CyberShieldNet Dev Server", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "CyberShieldNet API", "version": "1.0.0", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
