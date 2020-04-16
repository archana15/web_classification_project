import os
import random
from numpy import genfromtxt
import numpy as np 
import csv 
import pandas as pd 
import csv 

def load_data(path, cols_list):
    
    data = pd.read_csv(path,usecols=cols_list)

    train = (len(data)*80)//100
    train_data = data[:train]
   
    test_data = data[train:]
    
    train_label = []
    all_train_labels = train_data['category']
    for labels in all_train_labels:
        if labels not in train_label:
            train_label.append(labels)
    train_text = train_data['desc'] 
    # print(train_label)
    # print(train_text)

    test_label = []
    all_test_labels = test_data['category']
    for labels in all_test_labels:
        if labels not in test_label:
            test_label.append(labels)
    test_text = test_data['desc']
    # print(test_label)
    # print(test_text)

    return ((train_text,train_text),(test_text,test_label))

# columns = ['0','category','title','desc']
# load_data("/Users/archana/web_classification/test.csv", columns) 

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.take.html