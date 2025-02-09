from japan_ha.components.data_ingestion import DataIngestion
from japan_ha.components.data_validation import DataValidation
from japan_ha.exception.exception import JapanHeartAttackException
from japan_ha.logging.logger import logging
from japan_ha.entity.config_entity import DataIngestionConfig,DataValidationConfig
from japan_ha.entity.config_entity import TrainingPipelineConfig
from japan_ha.entity.artifacts_entity import DataIngestionArtifact,DataValidationArtifact
import sys

if __name__=="__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)

        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("data ingestion completed")
        print(dataingestionartifact)

        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)

        logging.info("initiate data validation ")
        datavalidationartifacts=data_validation.initiate_data_validation()
        logging.info("data validation completed")
        print(datavalidationartifacts)
        
    except Exception as e:
        raise JapanHeartAttackException(e,sys)
    