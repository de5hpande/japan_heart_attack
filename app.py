import sys
import os

import certifi
ca = certifi.where()

from japan_ha.exception.exception import JapanHeartAttackException
from japan_ha.logging.logger import logging
from japan_ha.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from japan_ha.utils.main_utils.utils import load_object

from japan_ha.utils.ml_utils.model.estimator import NetworkModel
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

from japan_ha.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from japan_ha.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise JapanHeartAttackException(e,sys)
    

@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File()):
    try:
        df=pd.read_csv(file.file)
        preprocessor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")

        network_model=NetworkModel(preprocessor=preprocessor,model=final_model)

        print(df.iloc[0])
        y_pred=network_model.predict(df)
        print(y_pred)
        df["predicted_column"]=y_pred
        print(df["predicted_column"])

        df.to_csv("prediction_output/output.csv")
        tabel_html=df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html",{"request":request,"table":tabel_html})
    except Exception as e:
        raise JapanHeartAttackException(e,sys)
    
# if __name__=="__main__":
#     app_run(app,host="localhost",port=8000)

if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)



