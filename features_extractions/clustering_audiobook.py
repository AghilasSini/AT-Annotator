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



# classifier information
from keras.layers import Input
from keras.models import Model
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



def clustering_with_hdbscan(data):
	clust_count = np.linspace(1, 20, num=20, dtype='int')

	clust_number = 2
	plot_number = 1
	plt.figure (figsize=(17,12))
	while clust_number < 21:
	    hdb = hdbscan.HDBSCAN(min_cluster_size=clust_number)
	    hdb_pred = hdb.fit(data)
	    plt.subplot(5, 4, plot_number, title = 'Min. Cluster Size = {}'.format(clust_number))
	    plt.scatter(unbal[:,0], unbal[:,1], c=hdb_pred.labels_, cmap=cmap)
	    plot_number += 1
	    clust_number += 1
	    
	plt.tight_layout()
	plt.show()



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
	clustering_with_hdbscan(train_vecs)

	# train_vecs['utterance']=df_.index.to_list()



	# # X_labeled['sentence_label']=sentence_emotion_labeling
	# cluster_='cluster_labels_13'
	# # cluster_labeling=X_labeled[['utterance','sentence_label',cluster_]].groupby(cluster_).size()
	
	# cluster_labeling=train_vecs[['utterance',cluster_]].groupby(cluster_).size()
	# print(cluster_labeling)
if __name__ == '__main__':
	main()