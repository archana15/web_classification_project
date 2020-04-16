from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
#from exploring_data import plot_sample_length_distribution
from loading_data import load_data

def sequence_vectorize(train_texts, val_texts):
    TOP_K = 20000

    # Limit on the length of text sequences. Sequences longer than this will be truncated.
    MAX_SEQUENCE_LENGTH = 500

    '''i/p ->
        train_texts: list, training text strings.
        test_texts: list, validation text strings.

    0/p ->
        x_train, x_val, word_index: vectorized training and validation
        texts and word index dictionary.'''
    
    #vocabulary
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts) 

    #vectorisation
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

columns = ['0','category','title','desc']
((train_text,train_text),(test_text,test_label)) = load_data("/Users/archana/web_classification/dmoz.csv", columns)
#no_rows,words_per_row = get_num_of_words_per_sample(train_text)
train, test, tokenizer = sequence_vectorize (train_text,test_text)
# print(train)
# print("--------------------------------------------------------------------")
print(test)
# print("---------------------------------------------------------------------")
# print(tokenizer)


    