from keras.layers import Dropout, Dense, GRU, Embedding,Activation
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from keras.layers.merge import Concatenate
import argparse
import glob
import os
import sys
import re
import pandas as pd
from keras.layers import LSTM
from sklearn.model_selection import train_test_split



def loadData_Tokenizer(X_train, X_test,MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=500):
	np.random.seed(7)
	text = np.concatenate((X_train, X_test), axis=0)
	text = np.array(text)
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(text)
	sequences = tokenizer.texts_to_sequences(text)
	word_index = tokenizer.word_index
	text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	print('Found %s unique tokens.' % len(word_index))
	indices = np.arange(text.shape[0])
	# np.random.shuffle(indices)
	text = text[indices]
	print(text.shape)
	X_train = text[0:len(X_train), ]
	X_test = text[len(X_train):, ]
	embeddings_index = {}
	f = open("/home/aghilas/Workspace/Experiments/SynPaFlex-Code/ml_template/classification/dataset/synpaflex_w2v.txt",encoding="utf8")
	for line in f:

		values = line.split()
		word = values[0]
		try:
			coefs = np.asarray(values[1:], dtype='float32')
		except:
			pass
		embeddings_index[word] = coefs
	f.close()
	print('Total %s word vectors.' % len(embeddings_index))
	return (X_train, X_test, word_index,embeddings_index)



def Build_Model_RNN_Text(word_index, embeddings_index, nclasses,  MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=100, dropout=0.5):
	"""
	def buildModel_RNN(word_index, embeddings_index, nclasses,  MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50, dropout=0.5):
	word_index in word index ,
	embeddings_index is embeddings index, look at data_helper.py
	nClasses is number of classes,
	MAX_SEQUENCE_LENGTH is maximum lenght of text sequences
	"""

	model = Sequential()
	hidden_layer = 4
	gru_node = 128

	embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			if len(embedding_matrix[i]) != len(embedding_vector):
				print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
					  "into shape", str(len(embedding_vector)), " Please make sure your"
																" EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
				exit(1)
			embedding_matrix[i] = embedding_vector
	model.add(Embedding(len(word_index) + 1,
								EMBEDDING_DIM,
								weights=[embedding_matrix],
								input_length=MAX_SEQUENCE_LENGTH,
								trainable=True))


	model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
	model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
	model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
	model.add(LSTM(gru_node, recurrent_dropout=0.2))
	model.add(Dense(1024,activation='relu'))
	model.add(Dense(nclasses))
	model.add(Activation('softmax'))
	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	return model




def clean_text(Text):
	Text = re.sub("\n", " ", Text)
	Text = re.sub("--", "", Text)
	Text = re.sub("\.\.\.", ".", Text)
	Text = Text.lower()
	# Text = re.split("[.!?]", Text)
	Sent = re.split("\W", Text)
	Sent = [Token for Token in Sent if Token]
	return Sent


def build_args():
	parser=argparse.ArgumentParser(description='')
	parser.add_argument('path', type=str, nargs=1, help='data input')
	return parser.parse_args()


def main():
	args=build_args()
	# 
	# newsgroups_train = fetch_20newsgroups(subset='train')
	# newsgroups_test = fetch_20newsgroups(subset='test')
	# X_train = newsgroups_train.data
	# X_test = newsgroups_test.data
	# y_train = newsgroups_train.target
	# y_test = newsgroups_test.target

	# 
	# Parsing arguments
	

	dataset=pd.read_csv(args.path[0],sep='\t')


	convert_dict={
	'label':'category',
			
	}
	# # print(cat_list)
	dataset = dataset.astype(convert_dict)
	dataset['label_cat'] = dataset.label.cat.codes
	data=[]
	for frame in dataset['text'].to_list():
		data.append(clean_text(frame))

	
	target_data=dataset['label_cat'].to_list()
	
	X_train, X_test, y_train, y_test  = train_test_split(data, target_data, test_size=0.3)



	X_train_Glove,X_test_Glove, word_index,embeddings_index = loadData_Tokenizer(X_train,X_test)


	model_RNN = Build_Model_RNN_Text(word_index,embeddings_index, 3)

	model_RNN.summary()

	model_RNN.fit(X_train_Glove, y_train,
								  validation_data=(X_test_Glove, y_test),
								  epochs=20,
								  batch_size=128,
								  verbose=2)

	predicted = model_RNN.predict_classes(X_test_Glove)

	predicted = np.argmax(predicted, axis=1)
	
	print(metrics.classification_report(y_test, predicted))


if __name__ == '__main__':
	main()