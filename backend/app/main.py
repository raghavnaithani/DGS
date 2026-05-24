from fastapi import FastAPI

app = FastAPI(title="DGS Backend")


@app.get("/health")
async def health():
    return {"status": "ok"}
