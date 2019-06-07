from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
import string

filename = 'input_afr.txt'
# save and load sequences to/from file
out_filename = inp_filename = 'afr.txt'
# organize into sequences of tokens
length = 50 + 1
sequences = []
epochs = 200

# load document and read text
def load_file(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

# turn a doc into clean tokens
def clean_file(doc):
	# remove punctuation from each token in doc
	translator = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(translator) for w in doc.split()]
	# remove remaining non-alphabetic tokens and transform into lowercase
	tokens = [w.lower() for w in tokens if w.isalpha()]
	return tokens

def make_sequences():
    for i in range(length, len(tokens)):
        # make sequence of tokens and convert into a line
        line = ' '.join(tokens[i-length:i])
        # store sequences in list
        sequences.append(line)

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def setup_tokenizer():
    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, sequences, vocab_size

def separate_inp_out():
    global sequences
    # separate into input and output
    sequences = array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)
    return X, y

def run_model():
    # define layers of model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=X.shape[1]))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.6))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    # compile and fit model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# save checkpoint if model has improved
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X, y, batch_size=128, epochs=epochs, callbacks=callbacks_list)
    return model

doc = load_file(filename)
tokens = clean_file(doc)
make_sequences()
save_doc(sequences, out_filename)
doc = load_file(inp_filename)
lines = doc.split('\n')
tokenizer, sequences, vocab_size = setup_tokenizer()
X, y = separate_inp_out()
model = run_model()
model.save("model_name.h5")
dump(tokenizer, open("tokenizer_name.pkl", 'wb'))
