from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import tensorflow as tf
import numpy as np

from building_model import sepcnn_model
from loading_data import load_data 
from vectorization import sequence_vectorize
from exploring_data import get_num_of_words_per_sample
from exploring_data import get_num_classes


FLAGS = None
TOP_K = 20000

def _get_embedding_matrix(word_index, embedding_data_dir, embedding_dim):
    """Gets embedding matrix from the embedding index data.
    # Arguments
        word_index: dict, word to index map that was generated from the data. 
        **o/p of vectorisation
        
        embedding_data_dir: string, path to the pre-training embeddings.
        ** 
        
        embedding_dim: int, dimension of the embedding vectors.
    # Returns
        dict, word vectors for words in word_index from pre-trained embedding.

    # Read the pre-trained embedding file and get word to word vector mappings.
    embedding_matrix_all = {}"""

    embedding_matrix_all = {}       

    # We are using 200d GloVe embeddings.
    fname = os.path.join(embedding_data_dir, 'glove.6B.200d.txt')
    with open(fname) as f:
        for line in f:  # Every line contains word followed by the vector value 
            # line = the -0.071549 0.093459 0.023738 -0.090339 0.056123 0.32547 -0.39796 -0.092139 0.061181 -0.1895 0.13061 0.14349 0.011479 0.38158 0.5403 -0.14088 0.24315 0.23036 -0.55339 0.048154 0.45662 3.2338 0.020199 0.049019 -0.014132 0.076017 -0.11527 0.2006 -0.077657 0.24328 0.16368 -0.34118 -0.06607 0.10152 0.038232 -0.17668 -0.88153 -0.33895 -0.035481 -0.55095 -0.016899 -0.43982 0.039004 0.40447 -0.2588 0.64594 0.26641 0.28009 -0.024625 0.63302 -0.317 0.10271 0.30886 0.097792 -0.38227 0.086552 0.047075 0.23511 -0.32127 -0.28538 0.1667 -0.0049707 -0.62714 -0.24904 0.29713 0.14379 -0.12325 -0.058178 -0.001029 -0.082126 0.36935 -0.00058442 0.34286 0.28426 -0.068599 0.65747 -0.029087 0.16184 0.073672 -0.30343 0.095733 -0.5286 -0.22898 0.064079 0.015218 0.34921 -0.4396 -0.43983 0.77515 -0.87767 -0.087504 0.39598 0.62362 -0.26211 -0.30539 -0.022964 0.30567 0.06766 0.15383 -0.11211 -0.09154 0.082562 0.16897 -0.032952 -0.28775 -0.2232 -0.090426 1.2407 -0.18244 -0.0075219 -0.041388 -0.011083 0.078186 0.38511 0.23334 0.14414 -0.0009107 -0.26388 -0.20481 0.10099 0.14076 0.28834 -0.045429 0.37247 0.13645 -0.67457 0.22786 0.12599 0.029091 0.030428 -0.13028 0.19408 0.49014 -0.39121 -0.075952 0.074731 0.18902 -0.16922 -0.26019 -0.039771 -0.24153 0.10875 0.30434 0.036009 1.4264 0.12759 -0.073811 -0.20418 0.0080016 0.15381 0.20223 0.28274 0.096206 -0.33634 0.50983 0.32625 -0.26535 0.374 -0.30388 -0.40033 -0.04291 -0.067897 -0.29332 0.10978 -0.045365 0.23222 -0.31134 -0.28983 -0.66687 0.53097 0.19461 0.3667 0.26185 -0.65187 0.10266 0.11363 -0.12953 -0.68246 -0.18751 0.1476 1.0765 -0.22908 -0.0093435 -0.20651 -0.35225 -0.2672 -0.0034307 0.25906 0.21759 0.66158 0.1218 0.19957 -0.20303 0.34474 -0.24328 0.13139 -0.0088767 0.33617 0.030591 0.25577
            values = line.split() # 
            word = values[0] # word = the
            coefs = np.asarray(values[1:], dtype='float32') #converts the values into float
            # coefs = [-0.071549, 0.093459, 0.023738, -0.090339, 0.056123 0.32547 -0.39796 -0.092139 0.061181 -0.1895 0.13061 0.14349 0.011479 0.38158 0.5403 -0.14088 0.24315 0.23036 -0.55339 0.048154 0.45662 3.2338 0.020199 0.049019 -0.014132 0.076017 -0.11527 0.2006 -0.077657 0.24328 0.16368 -0.34118 -0.06607 0.10152 0.038232 -0.17668 -0.88153 -0.33895 -0.035481 -0.55095 -0.016899 -0.43982 0.039004 0.40447 -0.2588 0.64594 0.26641 0.28009 -0.024625 0.63302 -0.317 0.10271 0.30886 0.097792 -0.38227 0.086552 0.047075 0.23511 -0.32127 -0.28538 0.1667 -0.0049707 -0.62714 -0.24904 0.29713 0.14379 -0.12325 -0.058178 -0.001029 -0.082126 0.36935 -0.00058442 0.34286 0.28426 -0.068599 0.65747 -0.029087 0.16184 0.073672 -0.30343 0.095733 -0.5286 -0.22898 0.064079 0.015218 0.34921 -0.4396 -0.43983 0.77515 -0.87767 -0.087504 0.39598 0.62362 -0.26211 -0.30539 -0.022964 0.30567 0.06766 0.15383 -0.11211 -0.09154 0.082562 0.16897 -0.032952 -0.28775 -0.2232 -0.090426 1.2407 -0.18244 -0.0075219 -0.041388 -0.011083 0.078186 0.38511 0.23334 0.14414 -0.0009107 -0.26388 -0.20481 0.10099 0.14076 0.28834 -0.045429 0.37247 0.13645 -0.67457 0.22786 0.12599 0.029091 0.030428 -0.13028 0.19408 0.49014 -0.39121 -0.075952 0.074731 0.18902 -0.16922 -0.26019 -0.039771 -0.24153 0.10875 0.30434 0.036009 1.4264 0.12759 -0.073811 -0.20418 0.0080016 0.15381 0.20223 0.28274 0.096206 -0.33634 0.50983 0.32625 -0.26535 0.374 -0.30388 -0.40033 -0.04291 -0.067897 -0.29332 0.10978 -0.045365 0.23222 -0.31134 -0.28983 -0.66687 0.53097 0.19461 0.3667 0.26185 -0.65187 0.10266 0.11363 -0.12953 -0.68246 -0.18751 0.1476 1.0765 -0.22908 -0.0093435 -0.20651 -0.35225 -0.2672 -0.0034307 0.25906 0.21759 0.66158 0.1218 0.19957 -0.20303 0.34474 -0.24328 0.13139 -0.0088767 0.33617 0.030591 0.25577
            embedding_matrix_all[word] = coefs # thi matrix will have all the values in matrics 

    # Prepare embedding matrix with just the words in our word_index dictionary
    num_words = min(len(word_index) + 1, TOP_K) # num_words = 20000
    embedding_matrix = np.zeros((num_words, embedding_dim)) # if words<20k it will befilled with zeros

    for word, i in word_index.items():
        if i >= TOP_K:
            continue
        embedding_vector = embedding_matrix_all.get(word) # embedding_vector = the,....
        if embedding_vector is not None: #true
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix # embedding_mtrix will contain all the words 

def train_fine_tuned_sequence_model(data,
                                    embedding_data_dir,
                                    learning_rate=1e-3,
                                    epochs=1000,
                                    batch_size=128,
                                    blocks=2,
                                    filters=64,
                                    dropout_rate=0.2,
                                    embedding_dim=200,
                                    kernel_size=3,
                                    pool_size=3):
    """Trains sequence model on the given dataset.
    # Arguments
        data: tuples of training and test texts and labels. - dmoz.csv 
        embedding_data_dir: string, path to the pre-training embeddings. - glove.6b.txt
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch. - at once(batch) how many samples to process
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
        
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = 14
    #num_classes = get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val, word_index = sequence_vectorize(train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K) # num_features = 20k

    embedding_matrix = _get_embedding_matrix(
        word_index, embedding_data_dir, embedding_dim) 
    # word_index - op of vectorizn, glove.txt, 200

    # Create model instance. First time we will train rest of network while
    # keeping embedding layer weights frozen. So, we set
    # is_embedding_trainable as False.
    model = sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features,
                                     use_pretrained_embedding=True,
                                     is_embedding_trainable=False,
                                     embedding_matrix=embedding_matrix)
    # this our model, this is the embedding layer

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy' #true
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    # fit method trains the model
    model.fit(x_train,
              train_labels,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=(x_val, val_labels),
              verbose=2,  # Logs once per epoch.
              batch_size=batch_size)

    # Save the model.
    model.save_weights('sequence_model_with_pre_trained_embedding.h5')

    # Create another model instance. This time we will unfreeze the embedding
    # layer and let it fine-tune to the given dataset.
    model = sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features,
                                     use_pretrained_embedding=True,
                                     is_embedding_trainable=True,
                                     embedding_matrix=embedding_matrix)

    # Compile model with learning parameters.
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Load the weights that we had saved into this new model.
    #the first model weights/output is used to train our actual model
    model.load_weights('sequence_model_with_pre_trained_embedding.h5')

    # Train and validate model.
    history = model.fit(x_train,
                        train_labels,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_val, val_labels),
                        verbose=2,  # Logs once per epoch.
                        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('sepcnn_fine_tuned_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='input data directory')
    parser.add_argument('--embedding_data_dir', type=str, default='./data',
                        help='embedding input data directory')
    FLAGS, unparsed = parser.parse_known_args()

    # Using the tweet weather topic classification dataset to demonstrate
    # training sequence model with fine-tuned pre-trained embedding.

    columns = ['0','category','title','desc']
    # ((train_text,train_text),(test_text,test_label)) 
    data = load_data("/Users/archana/web_classification/dmoz.csv", columns)
    train_fine_tuned_sequence_model(data, "/Users/archana/web_classification")
    
