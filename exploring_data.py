import numpy as np
import matplotlib.pyplot as plt
from loading_data import load_data 

#ploting
# x axis -> length of the rows 
# y axis ->  no of rows with same length 


def get_num_of_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    count = 0
    for i in sample_texts:
        count = count+1
    return (count,np.median(num_words))

def plot_sample_length_distribution(sample_texts):
    x = []
    for row in sample_texts :
        x.append(len(row))
    
    plt.hist(x, 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()
    
# columns = ['0','category','title','desc']
# ((train_text,train_text),(test_text,test_label)) = load_data("/Users/archana/web_classification/test.csv", columns)
# no_rows,words_per_row = get_num_of_words_per_sample(train_text)
# print(no_rows)
# print(words_per_row)
# print("ratio :")
# print(no_rows//words_per_row) 
#plot_sample_length_distribution(train_text)



