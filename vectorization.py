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

    max_length = len(max(x_train, key=len)) # if the text is more 500 words it gets truncated to 500 words 
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length) # if the length of the text is less than 500 the sequence is added with more words
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

columns = ['0','category','title','desc']
((train_text,train_labels),(test_text,test_labels))  = load_data("/Users/archana/web_classification/test.csv", columns)
print(sequence_vectorize(train_text, test_text))