import os
import sys

from japan_ha.exception.exception import JapanHeartAttackException
from japan_ha.logging.logger import logging
from japan_ha.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise JapanHeartAttackException(e,sys)
    
    def predict(self,x):
        try:
            x_transform=self.preprocessor.transform(x)
            y_hat=self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise JapanHeartAttackException(e,sys)