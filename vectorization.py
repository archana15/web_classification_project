from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

NGRAM_RANGE = (1, 2)
TOP_K = 20000
MAX_SEQUENCE_LENGTH = 500

def ngram_vectorize(train_texts, val_texts):
    
    