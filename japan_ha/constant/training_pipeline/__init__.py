import os
import sys
import pandas as pd
import numpy as np

"""
defining common constant variable for training pipeline
"""

TARGET_COLUMN="Heart_Attack_Occurrence"
PIPELINE_NAME:str="japanheartattack"
ARTIFACT_DIR:str="Artifacts"
FILE_NAME:str="japan_heart_attack_dataset.csv"
TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME:str="japan_heart_attack"
DATA_INGESTION_DATABASE_NAME:str="japandata"
DATA_INGESTION_DIR_NAME:str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str="feature_store"
DATA_INGESTION_INGESTED_DIR:str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2



