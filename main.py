from japan_ha.components.data_ingestion import DataIngestion
from japan_ha.components.data_validation import DataValidation
from japan_ha.components.data_transformation import DataTransformation
from japan_ha.exception.exception import JapanHeartAttackException
from japan_ha.logging.logger import logging
from japan_ha.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from japan_ha.entity.config_entity import TrainingPipelineConfig
from japan_ha.entity.artifacts_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
import sys

if __name__=="__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)

        logging.info("Initiate the data ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("data ingestion completed")
        print(data_ingestion_artifact)

        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(data_ingestion_artifact,data_validation_config)

        logging.info("initiate data validation ")
        data_validation_artifacts=data_validation.initiate_data_validation()
        logging.info("data validation completed")
        print(data_validation_artifacts)


        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        data_transformation=DataTransformation(data_validation_artifacts,data_transformation_config)

        logging.info("initiate data transformation ")
        data_transformation_artifacts=data_transformation.initiate_data_transformation()
        logging.info("data transformation completed")
        print(data_transformation_artifacts)
        
    except Exception as e:
        raise JapanHeartAttackException(e,sys)
    