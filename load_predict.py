from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import untokenize

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('sepcnn_fine_tuned_model.h5')

# model.compile(loss = 'sparse_categorical_crossentropy',
#             optimizer = 'Adam',
#             metrics = ['accuracy'])
TOP_K = 20000

MAX_SEQUENCE_LENGTH = 500

data = ["Chronology of animated movies, television programs, and short cartoons. Includes animation filmographies and a list of anime television series.",
"Dedicated to anthropomorphic characters. Fan art, images, profiles, and links.",
"Keep up with developments in online animation for all skill levels. Download tools, and seek inspiration from online work."
"Overcoming Information,Essay by Patricia Pisters on the animated image and its changing relationship with the cinematic image.",
"British cartoon, animation and comic strip creations - links, reviews and news from the UK."
"British cartoon, animation and comic strip creations - links, reviews and news from the UK.",
"Michael Crandol takes an exhaustive look at the history of animation and animators/visionaries like Max Fleisher, Walter Lantz, and Otto Messmer."]

tokenizer = text.Tokenizer() 
tokenizer.fit_on_texts(data) 
x_train = tokenizer.texts_to_sequences(data)

x_train = sequence.pad_sequences(x_train,maxlen=131)

prediction = model.predict(x_train)

for i in prediction :
    pred = untokenize.untokenize(i)

print(pred)


