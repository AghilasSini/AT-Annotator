from keras.layers import Input, Dense
from keras.models import Model
import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pandas as pd

from sklearn.preprocessing import label_binarize
import string
# nltk.download('conll2002')
flatten = lambda l: [item for sublist in l for item in sublist]

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import os 
import sys


from sklearn.preprocessing import LabelEncoder
from math import sqrt
from sklearn.metrics import mean_squared_error

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import argparse
import matplotlib.cm as cm

import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
# nltk.corpus.conll2002.fileids()

from tqdm import tqdm_notebook as tqdm
from tqdm import trange


from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import confusion_matrix


from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import scale


from gensim.models.word2vec import Word2Vec

import gensim
import random
from collections import OrderedDict


from sklearn.model_selection import KFold


# classifier information
from keras.layers import  Dropout, Dense
from keras.models import Sequential
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
LabeledSentence = gensim.models.doc2vec.LabeledSentence
import hdbscan


def model_ae(X_train,x_test,n=600,encoding_dim=2):
	# this is the size of our encoded representations
	encoding_dim = 2
	# http://gradientdescending.com/pca-vs-autoencoders-for-dimensionality-reduction/
	# r program


	# this is our input placeholder
	input = Input(shape=(n,))
	# "encoded" is the encoded representation of the input
	encoded = Dense(encoding_dim, activation='relu')(input)

	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(n, activation='sigmoid')(encoded)


	# this model maps an input to its reconstruction
	autoencoder = Model(input, decoded)



	# this model maps an input to its encoded representation
	encoder = Model(input, encoded)



	encoded_input = Input(shape=(encoding_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-1]
	# create the decoder model
	decoder = Model(encoded_input, decoder_layer(encoded_input))


	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	autoencoder.fit(X_train, X_train,
		epochs=50,
		batch_size=256,
		shuffle=True,
		validation_data=(x_test, x_test))
	predicted = encoder.predict(x_test)


	print(predicted)




	return autoencoder


def main():
	parser = argparse.ArgumentParser(description="")

	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")

	# Add arguments
	
	
	parser.add_argument("input_file", help="The input file to be projected")
	# parser.add_argument("speech_feats_file", help="The input file to be projected")
	# parser.add_argument("out_path_file", help="The input file to be projected")
	args = parser.parse_args()
	df_=pd.read_csv(args.input_file)
	print(df_.head())
	df_doc2vec=df_.copy()
	df_doc2vec=df_doc2vec.drop(['utterance'], axis=1)
	print(df_doc2vec.columns.to_list())

	# df_['sentence_label']=sentence_emotion_labeling

	print('loading the database')
	print(df_doc2vec.head())
	from sklearn.preprocessing import scale
	train_vecs = scale(df_doc2vec)
	print('scaling the data')

	## X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2)
	X_train,x_test,Y_train,y_test=train_test_split(train_vecs, df_['utterance'].to_list(),test_size=0.2)
	
	model=model_ae(X_train,x_test,train_vecs.shape[1],2)
	# print(model.summary())
	
	# model.fit(X_train, X_train,
		# epochs=50,
		# batch_size=256,
		# shuffle=True,
		# validation_data=(x_test, x_test))
	# predicted = model.predict(x_test)
	# print(predicted)
if __name__ == '__main__':
	main()