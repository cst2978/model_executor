from abc import ABC, abstractmethod
from model import loadmodel
from transform import loadtransform
import transform
from sklearn.feature_extraction import _stop_words as stop_words
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 
import string
import spacy
nlp = spacy.load('en_core_web_sm')

stopwords = stop_words.ENGLISH_STOP_WORDS
lemmatizer = WordNetLemmatizer()

class process(ABC):
    def __init__(
        self,
        model: loadmodel,
        transform: loadtransform
        ):

        self.model = model
        self.transform = transform

class runprocess(process):
    def clean(self, doc):
        text_no_namedentities = []
        document = nlp(doc)
        ents = [e.text for e in document.ents]
        for item in document:
            if item.text in ents:
                pass
            else:
                text_no_namedentities.append(item.text)
        doc = (" ".join(text_no_namedentities))

        doc = doc.lower().strip()
        doc = doc.replace("</br>", " ") 
        doc = doc.replace("-", " ") 
        doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
        doc = " ".join([token for token in doc.split() if token not in stopwords])    
        doc = "".join([lemmatizer.lemmatize(word) for word in doc])

        return doc
    
    def create_transformation(self, data):
        print ('2. Transformation Started')
        data['text'] = data['text'].apply(self.clean)
        self._doc = data
        self.docs = list(self._doc['text'])
        self._text_transform = self.transform(self.docs).vectorize()

    
    def process_final(self, data):
        self.create_transformation(data)

        return self.model().predict(self._text_transform)


