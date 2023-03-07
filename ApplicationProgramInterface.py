from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"Root": "You are visting Root"}