# afr-neural-folktales
This repository contains the code, data and models for the "Exploring West African Folk Narrative Texts using Machine Learning" master thesis.

The "data" folder stores the Western European corpora of folk tales needed to run the code.
The "code" folder contains five Python files and three Jupyter Notebook files, each belonging to one of the three experiments described in the thesis:

Experiment 1 - Text Generation:
- word_level_rnn.py: Trains a word-level RNN model for text generation.
- word_level_predictions.py: Generates text based on a model file ("weights.hdf5"), a file with sequences ("afr_sequences.txt") and a tokenizer ("tokenizer.pkl").
- char_level_rnn.py: Trains a character-level RNN model for text generation.
- char_level_predictions.py: Generates text based on a model file ("model.hdf5").

Experiment 2 - Text Classification:
- lstm_classifier.py: Trains and evaluates a deep learning LSTM classification model.
- bow_classifier.py: Trains and evaluates a Bag-of-Words classificaiton.
- tsne.py: Generates a T-SNE interactive visualization in HTML.

Experiment 3 - Narrative Structure Analysis:
- structure_classifier.ipynb: Trains multiple machine learning algorithms for narrative structure classification.


Some of the code is derived from the following online machine/deep learning tutorials:
https://machinelearningmastery.com/

https://www.kdnuggets.com/
