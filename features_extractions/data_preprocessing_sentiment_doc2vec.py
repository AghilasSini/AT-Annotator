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
from sklearn.cluster import MiniBatchKMeans

# def get_negative_word_index(document):
	# for iword,word in enumerate(document):
		# if word =='ne':
			# return iword


# train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
# test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
def text_emotion(df, column):
	'''
	Takes a DataFrame and a specified column of text and adds 10 columns to the
	DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
	column containing the value of the text in that emotions
	INPUT: DataFrame, string
	OUTPUT: the original DataFrame with ten new columns
	'''

	new_df = df.copy()

	filepath = ('/home/aghilas/Workspace/Corpora/SentimentAnalysis/FEEL.csv')
	emolex_df = pd.read_csv(filepath,
							sep=';')
	# print(emolex_df.head())


	for i,polarity in enumerate(emolex_df['polarity'].to_list()):
		if polarity=='negative':
			emolex_df.at[i,'negative']=1
			emolex_df.at[i,'positive']=0
		else:
			emolex_df.at[i,'negative']=0
			emolex_df.at[i,'positive']=1
	
	emotions = emolex_df.columns.drop(['word','polarity'])#'id','emotion_cat'])
	emo_df = pd.DataFrame(0, index=df.index, columns=emotions)
  
	
	with tqdm(total=len(list(new_df.iterrows()))) as pbar:
		for i, row in new_df.iterrows():
			pbar.update(1)
			text=new_df.loc[i][column]
			# print(text)
			document=[token for token in nltk.word_tokenize(text,language='french')]

			# negative_sent_word_index=''
			# if 'ne' in document:
				# negative_sent_word_index=get_negative_word_index(document)
			# print(document)
			for word in document:
				# word = stemmer.stem(word.lower())
				emo_score = emolex_df[emolex_df.word == word]
				if not emo_score.empty:
					if len(emo_score)<=1:    
						for emotion in list(emotions):
							# print(emotion)
							emo_df.at[i, emotion] += emo_score[emotion]
	new_df = pd.concat([new_df, emo_df], axis=1)

	return new_df


# def clustering_(X,df,range_n_clusters):
# 	for n_clusters in range_n_clusters:
# 		hdb = hdbscan.HDBSCAN(min_cluster_size=int(n_clusters))
# 		hdb_pred = hdb.fit(X)
# 		cluster_labels=hdb_pred.labels_
# 		labels="cluster_labels_{}".format(n_clusters)
# 		if not labels in df.keys():
# 			df[labels]=cluster_labels
# 		return df

def call_silhout_(X,df,range_n_clusters):
	#new_df=df.copy()
	for n_clusters in range_n_clusters:
		# Create a subplot with 1 row and 2 columns
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(18, 7)

		# The 1st subplot is the silhouette plot
		# The silhouette coefficient can range from -1, 1 but in this example all
		# lie within [-0.1, 1]
		ax1.set_xlim([-0.1, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

		# Initialize the clusterer with n_clusters value and a random generator
		# seed of 10 for reproducibility.
		clusterer = KMeans(n_clusters=n_clusters,init=, random_state=10)
		# from sklearn.mixture import GaussianMixture
		#Predict GMM cluster membership
		# clusterer = GaussianMixture(n_components=n_clusters, random_state=10)

		# from sklearn.cluster import AgglomerativeClustering
		# clusterer = AgglomerativeClustering(n_clusters=n_clusters)

		cluster_labels = clusterer.fit_predict(X)
		labels="cluster_labels_{}".format(n_clusters)
		if not labels in df.keys():
			df[labels]=cluster_labels
		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# clusters
		silhouette_avg = silhouette_score(X, cluster_labels)
		print("For n_clusters =", n_clusters,
			  "The average silhouette_score is :", silhouette_avg)

		# # Compute the silhouette scores for each sample
		# sample_silhouette_values = silhouette_samples(X, cluster_labels)

		# y_lower = 10
		# for i in range(n_clusters):
		# 	# Aggregate the silhouette scores for samples belonging to
		# 	# cluster i, and sort them
		# 	ith_cluster_silhouette_values = \
		# 		sample_silhouette_values[cluster_labels == i]

		# 	ith_cluster_silhouette_values.sort()

		# 	size_cluster_i = ith_cluster_silhouette_values.shape[0]
		# 	y_upper = y_lower + size_cluster_i

		# 	color = cm.nipy_spectral(float(i) / n_clusters)
		# 	ax1.fill_betweenx(np.arange(y_lower, y_upper),
		# 					  0, ith_cluster_silhouette_values,
		# 					  facecolor=color, edgecolor=color, alpha=0.7)

		# 	# Label the silhouette plots with their cluster numbers at the middle
		# 	ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

		# 	# Compute the new y_lower for next plot
		# 	y_lower = y_upper + 10  # 10 for the 0 samples

		# ax1.set_title("The silhouette plot for the various clusters.")
		# ax1.set_xlabel("The silhouette coefficient values")
		# ax1.set_ylabel("Cluster label")

		# # The vertical line for average silhouette score of all the values
		# ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		# ax1.set_yticks([])  # Clear the yaxis labels / ticks
		# ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

		# # 2nd Plot showing the actual clusters formed
		# colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
		# ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
		# 			c=colors, edgecolor='k')

		# # Labeling the clusters
		# centers = clusterer.cluster_centers_
		# # Draw white circles at cluster centers
		# ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
		# 			c="white", alpha=1, s=200, edgecolor='k')

		# for i, c in enumerate(centers):
		# 	ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
		# 				s=50, edgecolor='k')

		# ax2.set_title("The visualization of the clustered data.")
		# ax2.set_xlabel("Feature space for the 1st feature")
		# ax2.set_ylabel("Feature space for the 2nd feature")

		# plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
		# 			  "with n_clusters = %d" % n_clusters),
		# 			 fontsize=14, fontweight='bold')

	# plt.show()
	return df

























# print(train_sents[0])



def word2features(sent, i):
	word = sent[i][0]
	postag = sent[i][1]
	lemma = sent[i][2]

	features = {
		# 'bias': 1.0,
		'word.lower()': word.lower(),
		# 'word[-3:]': word[-3:],
		# 'word[-2:]': word[-2:],
		# 'word.isupper()': int(word.isupper()),
		# 'word.istitle()': int(word.istitle()),
		# 'word.isdigit()': int(word.isdigit()),
		'postag': str(postag),
		'lemma': lemma,
		# 'postag[:2]': postag[:2],
		# 'lemma[:2]': postag[:2],
	}
	# if i > 0:
	# 	word1 = sent[i-1][0]
	# 	postag1 = sent[i-1][1]
	# 	lemma1 = sent[i-1][2]
	# 	features.update({
	# 		'-1:word.lower()': word1.lower(),
	# 		'-1:word.istitle()': word1.istitle(),
	# 		'-1:word.isupper()': word1.isupper(),
	# 		'-1:postag': postag1,
	# 		'-1:lemma': lemma1,
	# 		'-1:postag[:2]': postag1[:2],
	# 		'-1:lemma[:2]': lemma1[:2],
	# 	})
	# else:
	# 	features['BOS'] = True


	# if i < len(sent)-1:
	# 	word1 = sent[i+1][0]
	# 	postag1 = sent[i+1][1]
	# 	lemma1 = sent[i+1][2]
	# 	features.update({
	# 		'+1:word.lower()': word1.lower(),
	# 		'+1:word.istitle()': word1.istitle(),
	# 		'+1:word.isupper()': word1.isupper(),
	# 		'+1:postag': postag1,
	# 		'+1:postag[:2]': postag1[:2],
	# 		'+1:lemma': lemma1,
	# 		'+1:lemma[:2]': lemma1[:2],
	# 	})

	# else:
	# 	features['EOS'] = True

	
	return features


def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token,postag,lemma,label in sent]


def sent2tokens(sent):
	return [token for token, postag,lemma, label in sent]


def TFIDF(X,MAX_NB_WORDS=40):
		vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
		X = vectorizer_x.fit_transform(X).toarray()
		print("tf-idf with",str(np.array(X).shape[1]),"features")
		return X


#Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

#Get training set vectors from our models
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized







def Build_Model_DNN_Text(shape, nClasses, dropout=0.25):
		"""
		buildModel_DNN_Tex(shape, nClasses,dropout)
		Build Deep neural networks Model for text classification
		Shape is input feature space
		nClasses is number of classes
		"""
		model = Sequential()
		node = 512 # number of nodes
		nLayers = 3 # number of  hidden layer
		model.add(Dense(node,input_dim=shape,activation='relu'))
		model.add(Dropout(dropout))
		for i in range(0,nLayers):
			model.add(Dense(node,input_dim=node,activation='relu'))
			model.add(Dropout(dropout))
		model.add(Dense(nClasses, activation='softmax'))


		model.compile(loss='sparse_categorical_crossentropy',
									optimizer='adam',
									metrics=['accuracy'])

		return model

















def main():
	parser = argparse.ArgumentParser(description="")

	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")

	# Add arguments
	
	
	parser.add_argument("input_file", help="The input file to be projected")
	parser.add_argument("speech_feats_file", help="The input file to be projected")
	# parser.add_argument("out_path_file", help="The input file to be projected")
	args = parser.parse_args()
	data=pd.read_csv(args.input_file,sep='\t')
	sentences_list=list(set(data['utterance'].to_list()))


	from sklearn.model_selection import train_test_split

	sentences=[]

	for sent in sentences_list:
		subdata=data[data['utterance']==sent]
		sentences.append([( getattr(row, "token"),getattr(row, "pos"),getattr(row, "lemma"),getattr(row, "utterance")) for index, row in subdata.iterrows()])

	X = [sent2features(s) for s in sentences]
	
	data=pd.DataFrame(flatten(X))
	target_colomn=data.columns.to_list()

	y = [sent2labels(s) for s in sentences]
	utts=pd.DataFrame(flatten(y),columns=['utterance'])
	
	df=pd.concat([data,utts], axis=1)
	
	utterance_size_group=df.groupby('utterance').size()


	NB_MAX_WORD=max(utterance_size_group)
	NB_FEAT_WORD=df.shape[1]
	
	columns=[ '{}_{}'.format(nvar,str(i).zfill(3))  for i in range(NB_MAX_WORD) for nvar in target_colomn ]
	index=list(set(df['utterance'].to_list()))
	df_ = pd.DataFrame(index=index, columns=columns)
	# df_ = df_.fillna(0) # with 0s rather than NaNs
	# data = np.array([np.arange(len(index))]*len(columns)).T
	# df_ = pd.DataFrame(data, index=index, columns=columns)
	# print(columns)
	for  iutt,uttname in enumerate(index):
		subdata=df[df.utterance==uttname]
		subdata=subdata.drop(['utterance'], axis=1)
		# print(subdata)
		# print(subdata.keys().to_list())
		utt=subdata.values.reshape((-1,len(subdata)*len(target_colomn)))
		# print(utt.shape[1])
		df_.at[uttname,:utt.shape[1]]=utt #subdata.values.reshape(-1,len(subdata)*NB_FEAT_WORD)
		
	df_ = df_.fillna(0)
	word_keys=['word.lower()_{}'.format(str(i).zfill(3)) for i in range(NB_MAX_WORD)]
	lemma_keys=['lemma_{}'.format(str(i).zfill(3)) for i in range(NB_MAX_WORD)]
	pos_keys=['postag_{}'.format(str(i).zfill(3)) for i in range(NB_MAX_WORD)]
	# print(word_keys)
	for idx in range(len(df_)):
		df_.ix[idx, 'text'] =' '.join([val for val in df_.iloc[idx][word_keys].to_list() if val !=0 ])
		df_.ix[idx, 'text_lemmatized'] =' '.join([val for val in df_.iloc[idx][lemma_keys].to_list() if val !=0 ])





	df_=text_emotion(df_, 'text_lemmatized')
	df_['word_count'] = df_['text_lemmatized'].apply(nltk.word_tokenize).apply(len)
	emotions=['joy','fear','sadness','anger','surprise','disgust','negative','positive']

	# # # print(max(df_['word_count'].to_list()),NB_MAX_WORD)
	for emotion in emotions:
		df_[emotion] = df_[emotion] / df_['word_count']


	# print(df_.head())


	X_clean=cleanText(df_['text'].to_list())
	unsup_reviews = labelizeReviews(X_clean, 'UNSUP')
	n_dim = 300
	#Initialize model and build vocab
	print('Initialize model and build vocab')
	#instantiate our DM and DBOW models
	model_dm = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=n_dim, sample=1e-3, negative=5, workers=3,epochs=15)
	
	model_dmm = gensim.models.Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
	model_dmm.build_vocab(unsup_reviews)



	#build vocab over all reviews
	model_dm.build_vocab(unsup_reviews)
	model_dm.train(unsup_reviews,epochs=model_dm.epochs,total_examples=model_dm.corpus_count)

	#Get training set vectors from our models
	
	x_doc2vec=OrderedDict()
	for utt,text in zip(index,df_['text'].to_list()):
		tokens = model_dm.infer_vector(text)
		x_doc2vec[utt]=tokens
	df_doc2vec=pd.DataFrame(x_doc2vec).T
	df_doc2vec.columns=[ 'doc2vec_{}'.format(str(i).zfill(3)) for i in range(n_dim)]
	
	df_doc2vec=pd.concat([df_doc2vec,df_[emotions]],axis=1)
	# df_doc2vec.to_csv('doc2vec_{}'.format(os.path.basename(args.input_file)))

	from sklearn.preprocessing import scale
	train_vecs = scale(df_doc2vec)

	# create label for each row
	sentence_emotion_labeling=[]
	for i,row  in df_[emotions].iterrows():
		label='_'.join([ '{}{}'.format(emotion,str(round(getattr(row,emotion),2))) for emotion in emotions if getattr(row,emotion)>0])
		if label=='':
			label='unknown'
		sentence_emotion_labeling.append(label)

	# sentence_emotion_labeling 
			


	# df_['sentence_label']=sentence_emotion_labeling



	



	#using pca as dimension reduction technique

	PCA_model = PCA(n_components=3, random_state=42)
	X_standard = PCA_model.fit_transform(train_vecs)*(-1)

	# Single VD
	# from numpy import array
	# from sklearn.decomposition import TruncatedSVD


	# X_standard = TruncatedSVD(n_components=2).fit_transform(train_vecs)

	# using T-distributed Stochastic Neighbor Embedding (T-SNE)
	# from sklearn.manifold import TSNE
	# X_standard = TSNE(n_components=3).fit_transform(train_vecs)

	# from sklearn.decomposition import NMF
	# X_standard = NMF(n_components=2).fit_transform(X_standard)

	# from sklearn import random_projection

	# X_standard = random_projection.GaussianRandomProjection(n_components=2).fit_transform(X_standard)





	# print(PCA_model.explained_variance_ratio_)
	# clustering
	range_n_clusters =np.arange(6,7,+1)
	# print(df_.shape)
	# X_labeled=clustering_(X_standard,df_,range_n_clusters)
	X_labeled=call_silhout_(X_standard,df_doc2vec,range_n_clusters)
	print(X_labeled.head())
	X_labeled['utterance']=index
	X_labeled.to_csv('doc2vec_{}'.format(os.path.basename(args.input_file)),index=False)
	# X_labeled['sentence_label']=sentence_emotion_labeling
	# cluster_='cluster_labels_5'
	# # cluster_labeling=X_labeled[['utterance','sentence_label','cluster_labels_4']].groupby('cluster_labels_4')
	# cluster_labeling=X_labeled[['utterance',cluster_]].groupby(cluster_)

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



	# # final_data['fold']=np.zeros((final_data.shape[0]))
	# # 
	# # for idx,(train_index, test_index) in enumerate(kf.split(final_data)):
	# # 	# 
	# # 	final_data.at[test_index,'fold']=idx
	# # 	# print(X[test_index])
	# # outFilename='./Doc2Vec/cluster_{}_doc2vec_with_emolex.csv'.format(os.path.basename(args.input_file))
	# # final_data.to_csv(outFilename,index=False)	



if __name__ == '__main__':
	main()
