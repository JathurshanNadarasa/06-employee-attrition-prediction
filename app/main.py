from pathlib import Path
import joblib
from fastapi import FastAPI
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


class EmployeeData(BaseModel):
    Age: int = Field(ge=18, le=100)
    BusinessTravel: Literal[
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]
    DailyRate: int = Field(ge=0)
    Department: Literal[
        "Sales",
        "Research & Development",
        "Human Resources"
    ]

    DistanceFromHome: int = Field(ge=0)
    Education: int = Field(ge=1, le=5)
    EducationField: Literal[
            "Human Resources",
            "Life Sciences",
            "Marketing",
            "Medical",
            "Other",
            "Technical Degree"
        ]
    EnvironmentSatisfaction: int = Field(ge=1, le=4)
    Gender: Literal["Male", "Female"]
    HourlyRate: int = Field(ge=0)
    JobInvolvement: int = Field(ge=1, le=4)
    JobLevel: int = Field(ge=1, le=5)
    JobRole: Literal[
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"
        ]
    JobSatisfaction: int = Field(ge=1, le=4)
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    MonthlyIncome: int = Field(ge=0)
    MonthlyRate: int = Field(ge=0)
    NumCompaniesWorked: int = Field(ge=0)
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int = Field(ge=0)
    PerformanceRating: int = Field(ge=1, le=4)
    RelationshipSatisfaction: int = Field(ge=1, le=4)
    StockOptionLevel: int = Field(ge=0, le=3)
    TotalWorkingYears: int = Field(ge=0)
    TrainingTimesLastYear: int = Field(ge=0)
    WorkLifeBalance: int = Field(ge=1, le=4)
    YearsAtCompany: int = Field(ge=0)
    YearsInCurrentRole: int = Field(ge=0)
    YearsSinceLastPromotion: int = Field(ge=0)
    YearsWithCurrManager: int = Field(ge=0)
    AgeGroup: Literal[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65"
        ]


#Locate the saved model relative to this file
BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_PATH = (
    BASE_DIR / 'notebooks' / 'models' / 'employee_attrition_rf.pkl'
)

#Load the trained pipeline
model = joblib.load(MODEL_PATH)

#create the fastApi application
app = FastAPI(
    title='Employee Atrrition Prediction API',
    description = 'Predict employee attrition using a trained Random forest model',
    version='1.0.0',
)

@app.get('/')
def home():
    return{
        'message':'Employee Attrition Prediction APT is running'
    }


@app.get('/health')
def health_check():
    return{
        'status':'healthy',
        'model_loaded':model is not None,
    }



@app.post("/predict")
def predict_attrition(employee: EmployeeData):
    # Convert input data into a DataFrame
    employee_df = pd.DataFrame([employee.model_dump()])

    # Predict attrition class
    prediction = int(model.predict(employee_df)[0])

    # Get probability of attrition
    probabilities = model.predict_proba(employee_df)[0]
    attrition_probability = float(probabilities[1])

    return {
        "attrition_prediction": (
            "Yes" if prediction == 1 else "No"
        ),
        "attrition_probability": round(
            attrition_probability, 4
        )
    }
