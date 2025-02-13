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