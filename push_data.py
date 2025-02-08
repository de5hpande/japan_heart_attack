import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")


import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo

from japan_ha.exception.exception import JapanHeartAttackException

class JapanDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise JapanHeartAttackException(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise JapanHeartAttackException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.records=records

            self.mongo_clinet=pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.mongo_clinet[database]
            self.collection=self.database[collection]

            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise JapanHeartAttackException(e,sys)

if __name__=='__main__':
    FILE_PATH="data\japan_heart_attack_dataset.csv"
    DATABASE="japandata"
    collection="japan_heart_attack"
    japanobj=JapanDataExtract()
    records=japanobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=japanobj.insert_data_mongodb(records,DATABASE,collection)
    print(no_of_records)