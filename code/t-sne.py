#!/usr/bin/env python
# coding: utf-8

from keras.preprocessing import sequence
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
from keras.layers.core import Dense, SpatialDropout1D
from matplotlib.widgets import Button
from matplotlib.text import Annotation
from adjustText import adjust_text
from string import punctuation
from nltk.corpus import stopwords
from numpy import array
from collections import Counter, OrderedDict
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import mpld3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.patheffects as PathEffects
import time
import numpy as np
import os
np.set_printoptions(suppress=True)
pd.set_option("display.max_colwidth", 10000)

filename_afr = 'input_afr.txt'
filename_eur = 'input_eur.txt'
TOP_WORDS = 6500
max_story_length = 500 
vocab = Counter()
max_story_length = 500 
batch_size = 25
colors = []        
size = []
mark = []

# load doc into memory
def load_doc(filename):
    with open(filename, 'r') as open_file:
        txt = open_file.read()
        items = txt.split('\n\n')
    storyList = split_text(items)
    return storyList

# split texts into separate lines
def split_text(stories):
    storyList = []
    for story in stories:
        storyList.extend(story.splitlines())
    storyList = [clean_doc(i) for i in storyList]
    return storyList

# turn a doc into clean tokens
def clean_doc(doc):
    # make document lowercase
    doc = doc.lower()
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
    return ' '.join(tokens)

def add_to_vocab(lst):
    for story in lst:
        vocab.update(story.split())
        
def select_top_words(TOP_WORDS):
    return vocab.most_common()[:6500]

def story_to_id(stories):
    vectors = []
    for r in stories:
        words = r.split(" ")
        vector = np.zeros(len(words))
        for t, word in enumerate(words):
            for i, token in enumerate(vocab):
                if word == token[0]:
                    vector[t] = i
        vectors.append(vector)
    return vectors

def run_model():
    global auc
    model = Sequential()
    model.add(Embedding(TOP_WORDS, 32, input_length=max_story_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))  
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print('Train...')
    model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=5,
          validation_split=0.2)
    # evaluate
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    y_true, y_pred = Y_test, model.predict(X_test).round()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = auc(fpr, tpr)
    print('Test Accuracy: %f' % (acc), 'Test AUC-score: %f' % (auc))
    return model
    
def create_truncated_model(trained_model):
    model = Sequential()
    model.add(Embedding(TOP_WORDS, 32, input_length=max_story_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    for i, layer in enumerate(model.layers):
        layer.set_weights(trained_model.layers[i].get_weights())
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def calc_tsne():
    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(hidden_features)
    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

    #Run T-SNE on the PCA features.
    tsne = TSNE(n_components=2, verbose = 3)
    tsne_results = tsne.fit_transform(pca_result)
    return tsne_results
    
def tsne_to_list():
    new_results = []
    results = tsne_results
    results = results.tolist()
    for i in results:
        i = tuple(i)
        new_results.append(i)
    return new_results

def removeZeros(test_list):
    new_X = []
    for i in test_list:
        new_X.append(np.trim_zeros(i))
    return new_X

def id_to_story(numList):
    small = []
    large = []
    for array in numList:
        for i in range(0,len(array)):
            for j, token in enumerate(vocab):
                if array[i] == j:
                    small.append(token[0])
        small = ' '.join(small)
        large.append(small)
        small = []
    return large

def wrap_by_word(s, n):
    '''returns a string where \\n is inserted between every n words'''
    a = s.split()
    ret = ''
    for i in range(0, len(a), n):
        ret += ' '.join(a[i:i+n]) + '\n'
    return ret

def create_df():
    df = pd.DataFrame(generated_data, columns=['x','y'])
    df["Prediction"] = Y_pred
    df['Predicted class'] = 1
    df["True class"] = 1
    df["Story"] = generated_labels
    df["Class"] = Y_test
    df["Color"] = 1
    for index, row in df.iterrows():
        if row['Class'] == 0.:
            df["Color"][index] = str("blue")
        elif row['Class'] == 1.:
            df["Color"][index] = str("red")
    for index, row in df.iterrows():
        if row['Class'] == 0.:
            df["True class"][index] = str("Western European")
        elif row['Class'] == 1.:
            df["True class"][index] = str("West African")
    for index, row in df.iterrows():
        if row['Prediction'] < 0.5:
            df["Predicted class"][index] = str("Western European")
        elif row['Prediction'] >= 0.5:
            df["Predicted class"][index] = str("West African")
    return df

# classify a story as european (0) or african (1)
def predict_words(story, vocab, model):
    # preprocess
    new_int = []
    word = story
    vector = []
    for i, token in enumerate(vocab):
        if story == token[0]:
            vector.append(i)
    new_int.append(vector)
    new_int = sequence.pad_sequences(new_int, maxlen=max_story_length)
    yhat = model.predict(new_int, verbose=0)
    s = yhat[0,0]
    def red(x):
        x = (x-0.5)*(1/0.5)
        return str(int(255*x))
    def blue(x):
        return red(1-x)
    if s >= 0.5:
        return "<span style='color: rgb("+red(s)+",0,0)'>" + str(word) + "</span>"
    else:
        return "<span style='color: rgb(0,0,"+blue(s)+")'>" + str(word) + "</span>"

def color_words(): 
    colored_words = []
    colored_sentences = []
    for index, row in df.iterrows():
        story_row = row['Story'].replace("\n", " \n ")
        for word in story_row.split(" "):
            if word != "\n":
                colored_words.append(predict_words(word, vocab, model))
            elif word == "\n":
                colored_words.append(word)
        colored_words = ' '.join(colored_words)
        colored_sentences.append(colored_words)
        colored_words = []
    return colored_sentences

def makeColorLst():
    for index, row in df.iterrows():
        colors.append(row['Color'])

def adjustPoints():
    N = len(df)
    for i in range(N):
        if df['Predicted class'][i] == df['True class'][i]:
            size.append(80)
            mark.append(str("none"))
        else:
            size.append(220)
            mark.append(str('black'))
                    
def make_tsne():
    # Define some CSS to control our custom labels
    N = len(df)
    css = """
    table
    {
      border-collapse: collapse;
      max-width: 100%;
    }
    th
    {
      color: #ffffff;
      background-color: #000000;
    }
    td
    {
      background-color: #D3D3D3;
      font-size: 15px;
      padding: 15px;
    }
    .wrapper:hover .span2 {
        color: red;
    }
    table, th, td
    {
      font-family:Arial, Helvetica, sans-serif;
      border: 1px solid black;
      text-align: left;
    }
    """

    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'),figsize=(20, 12))
    scatter = ax.scatter(df.x,
                         df.y,
                         c=colors,
                         s=size,
                         alpha=0.4,
                         cmap=plt.cm.jet,
                         edgecolors=mark)
    ax.grid(color='white', linestyle='solid')
    ax.set_title("T-SNE story classifier", size=30)
    labels = []
    for i in range(N):
        labels.append(str(df.iloc[[i], 2:6].T.to_html(escape=False)).replace('\\n','<br/>'))
    tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels=labels, voffset=10, hoffset=10, css=css)
    recs = []
    classes = ["Western European", "West African"]
    clrs = ["blue", "red"]
    for i in range(0,len(clrs)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=clrs[i]))
    plt.legend(recs,classes,loc=4)
    mpld3.plugins.connect(fig, tooltip)
    plt.autoscale()
    mpld3.show()
    
african_stories = load_doc(filename_afr)
european_stories = load_doc(filename_eur)
X = african_stories + european_stories
y = array([1 for _ in range(len(african_stories))] + [0 for _ in range(len(european_stories))])
add_to_vocab(X)
vocab = select_top_words(TOP_WORDS)
X = story_to_id(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.20)
X_train = sequence.pad_sequences(X_train, maxlen=max_story_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_story_length)
model = run_model()
truncated_model = create_truncated_model(model)
hidden_features = truncated_model.predict(X_test)
tsne_results = calc_tsne()
generated_data = tsne_to_list()
X_test_clean = removeZeros(X_test)
t_sne_lst = id_to_story(X_test_clean)
generated_labels = [wrap_by_word(i, 10) for i in t_sne_lst]
Y_pred = model.predict(X_test)
df = create_df()
df["Words"] = color_words()
df = df[['x', 'y', 'Prediction', 'Predicted class', 'True class', 'Words', 'Story','Class', 'Color']]
df.to_pickle("t-sne_df.csv")
makeColorLst()
adjustPoints()
make_tsne()
