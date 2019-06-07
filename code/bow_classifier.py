#!/usr/bin/env python
# coding: utf-8

# A Bag-of-Words Model for Predicting Geographical Origin of Folktales.

from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Activation
from numpy import array
from sklearn.metrics import roc_curve, auc
import pandas as pd 

filename_afr = 'input_afr.txt'
filename_eur = 'input_eur.txt'
#Balanced or imbalanced dataset (Balanced -> same amount of folk tales).
dataset = "imbalanced"
#Create empty vocabulary.
vocab = Counter()
epochs = 50

#Predict on new texts
stories = [('Foolish Anansi thought he could trick a fisherman into doing his work for him. "Let\'s go fishing," he suggested. "Very well," said the fisherman, who was clever and quite wise to Anansi\'s tricks. "I\'ll make the nets and you can get tired for me." "Wait," said Anansi, "I\'ll make the nets and you can get tired for me!" Anansi made nets as his friend pretended to be tired. They caught four fish. The fisherman said, "Anansi, you take these. I\'ll take tomorrow\'s catch. It might be bigger." Greedily imagining the next day\'s catch, Anansi said, "No, you take these and I\'ll take tomorrow\'s fish." But the next day, the nets were rotting away and no fish were caught.  The fisherman said, "Anansi, take these rotten nets to market. You can sell them for much money." When Anansi shouted, "Rotten nets for sale!" in the marketplace, people beat him with sticks. "Some partner you are," Anansi said to the fisherman as he rubbed his bruises.  "I took the beatings.  At least you could have taken the pain." Anansi never tried to trick the fisherman again!', 'West African')
,('Turtles used to live on the land, they say, until the time a clever turtle was caught by some hunters. They brought him to their village and placed the turtle before the Chief, who said, "How shall we cook him?" "You\'ll have to kill me first," said the turtle, "and take me out of this shell." "We\'ll break your shell with sticks," they said. "That\'ll never work," said the turtle, "Why don\'t you throw me in the water and drown me?!" "Excellent idea," said the Chief. They took the turtle to the river and threw him into the water to drown him. They were congratulating themselves on their success in drowning the turtle, when two little green eyes poked up in the water and the laughing turtle said, "Don\'t get those cooking pots out too fast, foolish people! As he swam away he said, "I think I\'ll spend most of my time from now on, safely in the water." It has been that way ever since!', 'West African')
,('A wolf, who was out searching for a meal, saw a goat feeding on grass on top of a high cliff. Wishing to get the goat to climb down from the rock and into his grasp, he called out to her. "Excuse me, dear Goat," he said in a friendly voice, "It is very dangerous for you to be at such a height. Do come down before you injure yourself. Besides, the grass is much greener and thicker down here. Take my advice, and please come down from that high cliff." But the goat knew too well of the wolf\'s intent. "You don\'t care if I injure myself or not. You don\'t care if I eat good grass or bad. What you care about is eating me."', 'Western European')
,('It was a windy afternoon - the trees sighing, the birds chirping, the frogs croaking, and the fowls cackling. There were clouds in the sky and empty tins and leaves were being blown about.', 'West African')
,('On a hill, a sheep that had no wool saw horses, one of them pulling a heavy wagon, one carrying a big load, and one carrying a man quickly. The sheep said to the horses: \"My heart pains me, seeing a man driving horses.\" The horses said: \"Listen, sheep, our hearts pain us when we see this: a man, the master, makes the wool of the sheep into a warm garment for himself. And the sheep has no wool.\" Having heard this, the sheep fled into the plain.', 'Western European')
,('When the animals heard this noise, they became very much afraid, and the started running away. They threw their tins away. Lion was in front and he ran very fast towards home.', 'West African')
,('"There is a stream about ten kilometres away." said Hare. "We can get some water from it. But there is a big Crab which frightens people who go there. If our strong brothers will try, we can get water presently."', 'West African')
,('Her parents did not think about it for long. "Birds of a feather, flock together," they thought, and gave their consent. So Fat Trina became Heinz\'s wife, and drove out both of the goats. Heinz now enjoyed life, having no work to rest from, but his own laziness. He went out with her only now and then, saying, "I\'m doing this so that afterwards I will enjoy resting more. Otherwise I shall lose all feeling for it."', 'Western European')
,('In later days two mighty swans have been seen to fly from the nest. A light shone far through the air, far over the lands of the earth; the swan, with the strong beating of his wings, scattered the twilight mists, and the starry sky was seen, and it was as if it came nearer to the earth. That was the swan Tycho Brahe. "Yes, then," you say; "but in our own days?"', 'Western European')
,('One fine day in winter some ants were busy drying their store of corn, which had got rather damp during a long spell of rain. Presently up came a grasshopper and begged them to spare her a few grains, "For," she said, "I\'m simply starving." The ants stopped work for a moment, though this was against their principles. "May we ask," said they, "what you were doing with yourself all last summer? Why didn\'t you collect a store of food for the winter?" "The fact is," replied the grasshopper, "I was so busy singing that I hadn\'t the time."', 'Western European')]


# load and read doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_story(doc, vocab):
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

# load all docs in a directory
def process_docs(filename, vocab):
    lines = list()
    # load and clean the doc
    doc = load_doc(filename)
    stories = doc.split('\n\n')
    for story in stories:
        line = doc_to_story(story, vocab)
        # add to list
        lines.append(line)
    return lines
    
# load doc and add to vocab
def add_docs_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)
    return tokens

def run_model():
    global auc
    #define network
    model = Sequential()
    #model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(30, input_shape=(X_test.shape[1],)))
    model.add(Dropout(0.6))
    # model.add(LeakyReLU(alpha=0.01))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_train, Y_train, epochs=epochs, verbose=0)
    # evaluate
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    y_true, y_pred = Y_test, model.predict(X_test).round()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = auc(fpr, tpr)
    print('Test Accuracy: %f' % (acc), 'Test AUC-score: %f' % (auc), "\n")
    return model

# classify a review as Western European (0) or West African (1)
def predict_background(story, vocab, tokenizer, model):
    # clean
    tokens = clean_doc(story)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='freq')
    # prediction
    yhat = model.predict(encoded, verbose=0)
    print(story, '\n')
    print("Prediction value: " + str("%.2f" % yhat) + ";")
    if round(yhat[0,0])==1.0:
        return "West African"
    elif round(yhat[0,0]) == 0.0:
        return "Western European"

# Prepare vocabulary and preprocess data.
add_docs_to_vocab(filename_afr, vocab)
add_docs_to_vocab(filename_eur, vocab)

african_stories = process_docs(filename_afr, vocab)
european_stories = process_docs(filename_eur, vocab)

# Choose either balanced or imbalanced dataset.
if dataset == "balanced":
    european_stories = european_stories[:len(african_stories)]
elif dataset == "imbalanced":
    pass

# Create the tokenizer.
tokenizer = Tokenizer()
# Fit the tokenizer on the documents
train_docs = african_stories + european_stories
tokenizer.fit_on_texts(train_docs)

# Encode dataset
X = tokenizer.texts_to_matrix(train_docs, mode='freq')
y = array([1 for _ in range(len(african_stories))] + [0 for _ in range(len(european_stories))])

# Split data and run model.
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.10, random_state = 42)
model = run_model()

# Story prediction
for story in stories:
    prediction = predict_background(story[0], vocab, tokenizer, model)
    if story[1] == prediction:
        print(prediction, "- Correct!", "\n\n")
    else:
        print(prediction, "- Wrong!", "\n\n")
