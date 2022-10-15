from abc import ABC, abstractmethod
from typing import Dict, Tuple
from config import modelConfig
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class loadtransform:
    def __init__(self, load_text) -> None:
        print ('a. Loading Pkl file for vectors')
        self._vectorpath = f"{modelConfig.model_dir}/vectorizer_v1.pkl"
        self.load_vector_object()
        self.text = load_text
    
    def load_vector_object(self) -> None:
        print ('a.1 . Loading Pkl file for vectors')
        self._vectorobj = pickle.load(open(self._vectorpath, 'rb'))#pd.read_pickle(self._vectorpath)
        # Create new tfidfVectorizer with old vocabulary
        self._tf_voc = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True,
                         vocabulary = self._vectorobj.vocabulary_)
        print ('here')

    def vectorize (self):
        print ('a.2 . Vector started')
        load_array = self._tf_voc.fit_transform(self.text).toarray()
        return load_array