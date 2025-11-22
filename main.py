from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load trained model
model = joblib.load("notebooks/iris_model.pkl")

app = FastAPI(
    title="Iris ML API",
    description="Prediction API with FastAPI + ML model",
    version="1.0",
)


# Request schema
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Response schema
class IrisResponse(BaseModel):
    class_name: str
    class_id: int


LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.post("/predict", response_model=IrisResponse)
def predict(req: IrisRequest):
    data = np.array(
        [req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]
    ).reshape(1, -1)

    pred = model.predict(data)[0]
    return IrisResponse(class_name=LABELS[pred], class_id=int(pred))
