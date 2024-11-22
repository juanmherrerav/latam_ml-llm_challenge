import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from challenge.model import DelayModel

# Declare the global variable
flight_delay_model = None




@asynccontextmanager
async def lifespan(app: FastAPI):
    global flight_delay_model  # Declare the global variable inside lifespan
    print("LOADING MODEL")
    # Load and train the ML model
    data = pd.read_csv(r"data/data.csv")
    flight_delay_model = DelayModel()
    features, target = flight_delay_model.preprocess(data=data, target_column="delay")
    flight_delay_model.fit(features=features, target=target)
    print("LOAD MODEL DONE")
    
    yield  # Keep the app running while the model is loaded
    flight_delay_model = None  # Clean up when the app shuts down
    print("UNLOADING MODEL")

app = FastAPI(lifespan=lifespan)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    global flight_delay_model  # Use the global variable here
    try:

        input_data = pd.DataFrame(data.get("flights"))  # Convert input to DataFrame
        print(input_data)
        print(type(input_data))
        print(type(data))
        if not validate_data(input_data):
            raise HTTPException(status_code=400, detail="Wrong or badformated data")
        pr_predict = pd.get_dummies(input_data, columns=['OPERA', 'TIPOVUELO','MES'])
        print("-"+str(type(pr_predict)))
        if flight_delay_model is not None:
            print("-"+type(flight_delay_model))
            print("+"+type(flight_delay_model))
            predictions = flight_delay_model.predict(pr_predict)  # Predict with the model
            print("Prediction"+predictions)
            return {"predictions": predictions}
        else:
            raise HTTPException(status_code=500,detail="Model not loaded")
    except HTTPException as e:
        raise HTTPException(status_code=400, detail="wrong data")
    except:
        print("Whew!", sys.exc_info()[0], "occurred.")
        print("Next input please.")
        print()



def validate_data(input_data: pd.DataFrame) -> bool:
    """
    Validates the input data to ensure it meets the following criteria:
    - Is not null.
    - Contains at least one member.
    - For each record:
      - The "MES" value must be an integer in the range [1, 12].
      - The "TIPOVUELO" value must be a string, either 'N' or 'I'.
      - The "OPERA" value must be a string and one of the allowed airline names.

    Args:
        input_data (pd.DataFrame): Input data to validate.

    Returns:
        bool: True if all criteria are met, False otherwise.
    """
    # Check if input_data is not null and contains at least one member
    if input_data is None or input_data.empty:
        return False
    print(type(input_data))
    # Define allowed values for TIPOVUELO and OPERA
    allowed_tipovuelo = {'N', 'I'}
    allowed_opera = {
        'Aerolineas Argentinas', 
        'Grupo LATAM', 
        'Latin American Wings', 
        'Sky Airline', 
        'Copa Air'
    }

    # Iterate over the records and validate each row
    for index, row in input_data.iterrows():

        # Validate MES: integer in range [1, 12]
        if not (isinstance(row.get("MES"), int) and 1 <= row["MES"] <= 12):
            print("1")
            return False
        
        # Validate TIPOVUELO: string and in allowed values
        if not (isinstance(row.get("TIPOVUELO"), str) and row["TIPOVUELO"] in allowed_tipovuelo):
            print("1")
            return False
        
        # Validate OPERA: string and in allowed values
        if not (isinstance(row.get("OPERA"), str) and row["OPERA"] in allowed_opera):
            print("1")
            return False

    return True
