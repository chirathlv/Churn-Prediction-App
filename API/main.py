# Import Packages
import joblib # Model Elements
import uvicorn # Testing
import pandas as pd # Dataframe
from pydantic import BaseModel # Data Validation
from fastapi import FastAPI # API Library
import numpy as np

#--------------------------#

# Instantiate an API Instance
app = FastAPI(title="Churn Prediction API", version='1.0')

#--------------------------#

# Load Model Elements
le = joblib.load('../Model/label_encoder.joblib')
std_scaler = joblib.load('../Model/std_scaler.joblib')
categorical_features = joblib.load('../Model/categorical_features.joblib')
numerical_features = joblib.load('../Model/numerical_features.joblib')
xgb_clf = joblib.load('../Model/xgb_clf.joblib')

#--------------------------#

'''
Data Validation
1. Read the request body as JSON
2. Validate if the data has correct types
'''

class Data(BaseModel):
    city : str
    gender : str
    senior_citizen : str
    partner : str
    dependents : str
    phone_service : str
    multiple_lines : str
    internet_service : str
    online_security : str
    online_backup : str
    device_protection : str
    tech_support : str
    streaming_tv : str
    streaming_movies : str
    contract : str
    paperless_billing : str
    payment_method : str
    monthly_charges : float
    total_charges : float
    tenure : int
    lat : float
    long : float
    zip : int

#--------------------------#

# API End point
@app.get('/')
@app.get('/home')
def status():
    # API Status
    return {'message' : 'System is up and running'}

#--------------------------#

# Churn Prediction End point receives requests from the client
@app.post('/predict/')

async def predict(data : Data):
    # fetch data and create dataframe
    data_dict = data.dict()

    churn_df = pd.DataFrame.from_dict([data_dict])

    # Label Encoding
    churn_df[categorical_features] = le.transform(churn_df[categorical_features])

    # Standard Scaling
    churn_df[categorical_features+numerical_features] = std_scaler.transform(churn_df[categorical_features+numerical_features])
    
    # Prediction
    y_pred = xgb_clf.predict(churn_df)
    
    # Churn Label
    churn_label = ['Churn Customer' if i==1 else 'Staying Customer' for i in y_pred.tolist()]

    return {'prediction' : churn_label}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
     

