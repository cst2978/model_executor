from abc import ABC, abstractmethod
from typing import Dict, Tuple
from config import modelConfig
import pandas as pd
class loadmodel:
    def __init__(self):
        print ('b.1 . Loading Pkl file for Model')
        self._modelpath = f"{modelConfig.model_dir}/decisiontree.pkl"
        self.load_model_object()
    
    def load_model_object(self) -> None:
        print ('b.2 . Loading Pkl file for Model')
        self._model = pd.read_pickle(self._modelpath)

    def predict (self, load_array):
        print ('b.3 . Model Prediction started')
        prediction = self._model.predict(load_array)
        return prediction