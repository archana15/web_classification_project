import os
import random
from numpy import genfromtxt
import numpy as np 
import csv 
import pandas as pd 
import csv 
from sklearn.utils import shuffle


def load_data(path, cols_list):
    
    datas = pd.read_csv(path,usecols=cols_list)
    
    data = df = shuffle(datas)
    train = (len(data)*80)//100
    
    train_data = data[:train]
    test_data = data[train:]

    
    train_lab = train_data['category'] 
    dictionary = {'Arts':1, 
                        'Business':2, 
                        'Computers':3, 
                        'Games':4, 
                        'Health':5, 
                        'Home':6, 
                        'News':7, 
                        'Recreation':8, 
                        'Reference':9, 
                        'Science':10, 
                        'Shopping':11,
                        'Shopping':12, 
                        'Society':13}
    trn_labels = []
    for label in train_lab:
        trn_labels.append(dictionary[label])
    train_labels = np.asarray(trn_labels)

    train_texts = train_data['desc']

    test_lab = test_data['category']
    tst_labels = []
    for label in test_lab:
        tst_labels.append(dictionary[label])
    test_labels = np.asarray(tst_labels)

    test_texts = test_data['desc']

    return ((train_texts,train_labels),(test_texts, test_labels))

# columns = ['0','category','title','desc']
# load_data("/Users/archana/web_classification/dmoz.csv", columns) 
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.take.html