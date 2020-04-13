import os
import random
from numpy import genfromtxt
import numpy as np 
import csv 
import pandas as pd 

def load_data(path):

    csv = pd.read_csv(path)
    train = (len(csv)*80)//100     
    train_text = csv[:train] #desc
    train_label = #category 

    test_text = csv[train:]
    test_label = 

    return train_text, test_text


    # print(train_text)
    # print(test_text)


load_data("/Users/archana/iop_tech/test.csv")   
 