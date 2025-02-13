from japan_ha.constant.training_pipeline import TARGET_COLUMN
from japan_ha.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from japan_ha.entity.artifacts_entity import (
   DataTransformationArtifact, DataValidationArtifact
)
import sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline  # Use imblearn's Pipeline for SMOTE
from imblearn.over_sampling import SMOTE
from japan_ha.entity.config_entity import DataTransformationConfig
from japan_ha.exception.exception import JapanHeartAttackException
from japan_ha.logging.logger import logging
from japan_ha.utils.main_utils.utils import save_numpy_array_data,save_object
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline  # Use imblearn's Pipeline for SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise JapanHeartAttackException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise JapanHeartAttackException(e,sys)

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_columns =  ['Age', 'Cholesterol_Level','Stress_Levels', 'BMI','Heart_Rate','Systolic_BP','Diastolic_BP']
            categorical_columns =  ['Gender', 'Region', 'Smoking_History', 'Diabetes_History',
                                    'Hypertension_History', 'Diet_Quality', 'Alcohol_Consumption', 'Family_History'
                                    ,"Physical_Activity",]

            # Numerical pipeline with KNNImputer and StandardScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),  # Use KNNImputer instead of SimpleImputer
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline with KNNImputer, OneHotEncoder, and StandardScaler
            cat_pipeline = Pipeline(
            steps=[
                 ("imputer", SimpleImputer(strategy="most_frequent")),  # Use SimpleImputer for categorical data
                 ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                 ("scaler", StandardScaler(with_mean=False))
                                                                                 ]
                                                                                                    )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            # Add SMOTE to the pipeline using imblearn's Pipeline
            processor = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),  # Apply preprocessing
                    ("smote", SMOTE(sampling_strategy="minority"))  # Apply SMOTE
                ]
            )

            return processor

        except Exception as e:
            raise JapanHeartAttackException(e, sys)


        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of data transformation class")
        try:
            logging.info("starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)


            ##training data frame
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]


            ##testing dataframe
            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()

            # Convert 'Yes' → 1 and 'No' → 0
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            

            preprocessor=self.get_data_transformer_object()
            
            # transformed_input_train_feature=preprocessor.fit_resample(input_feature_train_df)
            # transformed_input_test_feature=preprocessor.fit_resample(input_feature_test_df)

            transformed_input_train_feature, target_feature_train_df = preprocessor.fit_resample(input_feature_train_df, target_feature_train_df)
            print(target_feature_train_df)
            # Fit and resample the preprocessor on the test data
            transformed_input_test_feature, target_feature_test_df = preprocessor.fit_resample(input_feature_test_df, target_feature_test_df)
            print(target_feature_test_df)
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]
            print(train_arr)
            print(test_arr)

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor,)

            save_object("final_model/preprocessor.pkl",preprocessor)

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


        except Exception as e:
            raise JapanHeartAttackException(e,sys)