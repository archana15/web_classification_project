from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

NGRAM_RANGE = (1, 2)
TOP_K = 20000
TOKEN_MODE = 'word'

def ngram_vectorize(train_texts, train_labels, val_texts):
    