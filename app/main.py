import pymongo
from random import randrange
from pymongo import MongoClient
import pandas as pd
import load
import nltk
from core import runprocess
import pandas as pd
from model import loadmodel
from transform import loadtransform
from config import modelConfig

CONNECTION_STRING = modelConfig.connection 
#read pickel
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
client = MongoClient(CONNECTION_STRING)

def read_data():
   # database
   db = client['ml']
   #Collection
   col = db['mlai']
   x = col.find()
   data = pd.DataFrame(list(x))
   data = data.head()
   result = runprocess(loadmodel,loadtransform).process_final(data) #Prediction on the model
    #print ({ "_id": data['_id'], "pred_result": result})
   data['preds'] = result
   print (data)
   #data_op = data[['_id','preds']].to_dict('records')

   return data[['_unit_id','preds']].to_dict('records')


def load_data(df):
    db = client['ml']
    mycol = db['mlai_op_test2']
    x = mycol.insert_many(df)
    return x

  
# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":   
  
   # Read the database
    df = read_data()
    
    x = load_data(df)
