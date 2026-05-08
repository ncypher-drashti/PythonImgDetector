from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
def predict(value: int):
    return {"result": value * 2}