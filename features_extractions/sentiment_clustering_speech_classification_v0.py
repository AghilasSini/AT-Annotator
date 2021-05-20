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

import hdbscan
from sklearn.cluster import MiniBatchKMeans

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def model_ae(X_train,x_test,n=300,encoding_dim=32):

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
		epochs=20,
		batch_size=32,
		shuffle=True,
		validation_data=(x_test, x_test))




	return encoder
















def call_silhout_(X,df,range_n_clusters):

	hyper_parm_turning=OrderedDict()
	for n_clusters in range_n_clusters:
	

		# Initialize the clusterer with n_clusters value and a random generator
		# seed of 10 for reproducibility.
		# clusterer = MiniBatchKMeans(n_clusters=n_clusters,init='k-means++', random_state=10)
		from sklearn.mixture import GaussianMixture
		# Predict GMM cluster membership
		clusterer = GaussianMixture(n_components=n_clusters, random_state=10)

		# from sklearn.cluster import AgglomerativeClustering
		# clusterer = AgglomerativeClustering(n_clusters=n_clusters)

		cluster_labels = clusterer.fit_predict(X)
		labels="cluster_labels_{}".format(n_clusters)
		if not labels in df.keys():
			df[labels]=cluster_labels


		sample_dist_std=np.std(df.groupby(labels).size())
		sample_dist_avrg=np.median(df.groupby(labels).size())


		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# clusters
		silhouette_avg = silhouette_score(X, cluster_labels)
		

		if not 'n_clusters' in hyper_parm_turning.keys():
			hyper_parm_turning['n_clusters']=[n_clusters]
		else:
			hyper_parm_turning['n_clusters'].append(n_clusters)

		if not  'silhouette_avg' in hyper_parm_turning.keys():
			hyper_parm_turning['silhouette_avg']=[silhouette_avg]
		else:
			hyper_parm_turning['silhouette_avg'].append(silhouette_avg)


		if not  'sample_dist_std' in hyper_parm_turning.keys():
			hyper_parm_turning['sample_dist_std']=[sample_dist_std]
		else:
			hyper_parm_turning['sample_dist_std'].append(sample_dist_std)



		if not 'sample_dist_avrg'  in hyper_parm_turning.keys():
			hyper_parm_turning['sample_dist_avrg']=[sample_dist_avrg]
		else:
			hyper_parm_turning['sample_dist_avrg'].append(sample_dist_avrg)

		print("For n_clusters =", n_clusters,
			  "The average silhouette_score is :", silhouette_avg)

	
	return df,hyper_parm_turning































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
	# print(df_.head())
	df_doc2vec=df_.copy()
	df_doc2vec=df_doc2vec.drop(['utterance'], axis=1)
	# print(df_doc2vec.columns.to_list())

	# df_['sentence_label']=sentence_emotion_labeling

	
	df_doc2vec = df_doc2vec[df_doc2vec.columns[:300]]
	print('loading the database')
	# print(df_doc2vec.head())
	print(df_doc2vec.shape)
	from sklearn.preprocessing import scale
	train_vecs = scale(df_doc2vec)
	print('scaling the data')


	#using pca as dimension reduction technique


	PCA_model = PCA(.90, random_state=42)
	X_standard = PCA_model.fit_transform(train_vecs)*(-1)
	print(X_standard.shape)
	# Single VD
	# from numpy import array
	# from sklearn.decomposition import TruncatedSVD

	# TruncatedSVD_model=TruncatedSVD(n_components=3)
	# X_standard = TruncatedSVD_model.fit_transform(train_vecs)




	# using T-distributed Stochastic Neighbor Embedding (T-SNE)
	# from sklearn.manifold import TSNE
	# X_standard = TSNE(n_components=3).fit_transform(train_vecs)

	# from sklearn.decomposition import NMF
	# NMF_model=NMF(n_components=3)
	# X_standard = NMF_model.fit_transform(train_vecs)

	# from sklearn import random_projection

	# X_standard = random_projection.GaussianRandomProjection(n_components=2).fit_transform(X_standard)

	# X_train,x_test,Y_train,y_test=train_test_split(train_vecs, df_['utterance'].to_list(),test_size=0.2)
	

	# encodeing=model_ae(X_train,x_test)
	# X_standard=scale(encodeing.predict(train_vecs))



	# print(X_standard)

	# print(PCA_model.explained_variance_ratio_)
	# print(TruncatedSVD_model.explained_variance_ratio_)
	# print(NMF_model.explained_variance_ratio_)
	# clustering
	range_n_clusters =np.arange(20,22,+1)
	# # print(df_.shape)
	X_labeled,hyper_parm_turning=call_silhout_(X_standard,df_,range_n_clusters)
	
	# print(X_labeled.head())
	X_labeled['utterance']=df_.index.to_list()
	# # X_labeled['sentence_label']=sentence_emotion_labeling
	cluster_='cluster_labels_20'
	# cluster_labeling=X_labeled[['utterance','sentence_label',cluster_]].groupby(cluster_).size()
	
	cluster_labeling=X_labeled[['utterance',cluster_]].groupby(cluster_).size()
	print(cluster_labeling)

	hyper_parm_turning=pd.DataFrame(hyper_parm_turning)
	# Sort the rows of dataframe by column 'Name'
	hyper_parm_turning = hyper_parm_turning.sort_values(by =['silhouette_avg','sample_dist_std'],ascending=False)
	 
	print("Contents of Sorted Dataframe based on a single column 'silhouette_avg' & 'sample_dist_std' : ")
	print(hyper_parm_turning)

	# print(hyper_parm_turning)
	# cluster=''
	
	# outPutData=OrderedDict()

	# for idx,group in cluster_labeling:
		
	# 	if cluster!=group[cluster_].to_list()[0] and group.shape[0]>80 :
	# 		cluster=group[cluster_].to_list()[0]
	# 		print('the shape of the group {} cluster name {}'.format(group.shape,cluster))
	# 		# print(group['utterance'].to_list())
	# 		# with codecs.open('./Doc2Vec/cluster_{}_doc2vec_with_emolex.scp'.format(cluster),'w','utf-8') as cluster:
	# 		# for utt,label in zip(group['utterance'].to_list(),group['sentence_label'].to_list()):
	# 		for utt in group['utterance'].to_list():#,group['sentence_label'].to_list()):
	# 			if not 'utterance' in outPutData.keys():
	# 				outPutData['utterance']=[utt]
	# 			else:
	# 				outPutData['utterance'].append(utt)

	# 			# if not 'emotion_label' in outPutData.keys():
	# 			# 	outPutData['emotion_label']=[label]
	# 			# else:
	# 			# 	outPutData['emotion_label'].append(label)

	# 			if not 'cluster' in outPutData.keys():
	# 				outPutData['cluster']=[cluster]
	# 			else:
	# 				outPutData['cluster'].append(cluster)

	# final_data=pd.DataFrame(outPutData)

	# speech_features=pd.read_csv(args.speech_feats_file)
	# # print(speech_features['utterance'].to_list())
	
	# # data_with_feat=speech_features.copy()
	# features=speech_features.columns.drop('utterance')
	# feat_data = pd.DataFrame(0, index=final_data.index, columns=features)
	# for i,row in final_data.iterrows():
	# 	utterance=getattr(row,'utterance')
	# 	feats = speech_features[speech_features.utterance == utterance]
	# 	for feat in list(features):
	# 		feat_data.at[i,feat]=feats[feat]
	# final_data = pd.concat([final_data, feat_data], axis=1)

	# convert_dict={
	# 'cluster':'category',
			
	# }
	# # # print(cat_list)
	# final_data = final_data.astype(convert_dict)

	# final_data['cluster_cat'] = final_data.cluster.cat.codes
	# print(final_data.head())
	

	# kf = KFold(n_splits=5,shuffle=True)
	# X=final_data[features].values
	# Y=final_data['cluster_cat'].values 
	# X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2)

	# # build the model
	# n_classes=len(set(Y))
	# print(n_classes)
	# model_DNN=Build_Model_DNN_Text(X_train.shape[1],n_classes)
	# print(model_DNN.summary())
	# cross_fold_accuracy=[]
	# for idx,(train_index, test_index) in enumerate(kf.split(X_train)):
	# 	# print("TRAIN:", train_index, "TEST:", test_index)

	# 	x_train=X_train[train_index]
	# 	x_eval=X_train[test_index]


	# 	y_train=Y_train[train_index]
	# 	y_eval=Y_train[test_index]
	# 	model_DNN.fit(x_train, y_train,validation_data=(x_eval, y_eval),
	# 															epochs=20,
	# 															batch_size=16,
	# 															verbose=2)

	# 	predicted = model_DNN.predict(x_test)
	# 	predicted = np.argmax(predicted, axis=1)
	# 	acc=accuracy_score(y_test,predicted)
	# 	cross_fold_accuracy.append(acc)
	# 	print('fold {} accuracy {}'.format(idx+1,acc*100))
	# print('cross folds acc {} (+/-{})'.format(np.mean(cross_fold_accuracy)*100,np.std(cross_fold_accuracy)*100))



	# final_data['fold']=np.zeros((final_data.shape[0]))
	# 
	# for idx,(train_index, test_index) in enumerate(kf.split(final_data)):
	# 	# 
	# 	final_data.at[test_index,'fold']=idx
	# 	# print(X[test_index])
	# outFilename='./Doc2Vec/cluster_{}_doc2vec_with_emolex.csv'.format(os.path.basename(args.input_file))
	# final_data.to_csv(outFilename,index=False)	



if __name__ == '__main__':
	main()
