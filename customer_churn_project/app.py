from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import pandas as pd
import joblib


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0"
)


MODEL_PATH = Path(__file__).parent / "model" / "churn_model.pkl"

model = joblib.load(MODEL_PATH)


class CustomerData(BaseModel):

    gender: Literal["Female", "Male"]

    SeniorCitizen: int = Field(..., ge=0, le=1)

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    tenure: int = Field(..., ge=0)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False
    )

    TotalCharges: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["TotalServices"] = (
        df[service_columns] == "Yes"
    ).sum(axis=1)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, float("inf")],
        labels=[
            "0-12",
            "13-24",
            "25-48",
            "49-60",
            "61+"
        ]
    )

    return df


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    try:

        customer_df = pd.DataFrame(
            [customer.model_dump()]
        )

        customer_df = engineer_features(customer_df)

        prediction = model.predict(customer_df)[0]

        probabilities = model.predict_proba(
            customer_df
        )[0]

        classes = model.classes_

        churn_index = list(classes).index("Yes")

        churn_probability = probabilities[
            churn_index
        ]

        return {
            "prediction": str(prediction),
            "churn_probability": round(
                float(churn_probability),
                4
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )