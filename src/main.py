from fastapi import FastAPI
from src.api.endpoints import predictions  # Updated import path

app = FastAPI()

app.include_router(predictions.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Flower Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)