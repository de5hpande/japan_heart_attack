from japan_ha.components.data_ingestion import DataIngestion
from japan_ha.exception.exception import JapanHeartAttackException
from japan_ha.logging.logger import logging
from japan_ha.entity.config_entity import DataIngestionConfig
from japan_ha.entity.config_entity import TrainingPipelineConfig
import sys

if __name__=="__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)

        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        
    except Exception as e:
        raise JapanHeartAttackException(e,sys)
    