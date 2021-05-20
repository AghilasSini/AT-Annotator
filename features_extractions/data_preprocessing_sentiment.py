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

import hdbscan
from sklearn.cluster import MiniBatchKMeans
from collections import OrderedDict

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



def call_silhout_(X,df,range_n_clusters):
	hyper_parm_turning=OrderedDict()		
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
		# clusterer = MiniBatchKMeans(n_clusters=n_clusters,init='k-means++', random_state=10)
		# clusterer=clusterer = GaussianMixture(n_components=n_clusters, random_state=10)
		from sklearn.mixture import GaussianMixture
		# Predict GMM cluster membership
		clusterer = GaussianMixture(n_components=n_clusters, random_state=10)
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
		print("For n_clusters =", n_clusters,
			  "The average silhouette_score is :", silhouette_avg)
		


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
		vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS,ngram_range=(1, 3),stop_words=nltk.corpus.stopwords.words('french'))
		X = vectorizer_x.fit_transform(X).toarray()
		print("tf-idf with",str(np.array(X).shape[1]),"features")
		return X
def cleanTxt(tokens):
	words=[]
	for token in tokens:
		if not token in string.punctuation:
			words.append(token.lower())
		else:
			continue
	return ' '.join(words)

def main():
	parser = argparse.ArgumentParser(description="")

	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")

	# Add arguments
	
	
	parser.add_argument("input_file", help="The input file to be projected")
	# parser.add_argument("out_path_file", help="The input file to be projected")
	args = parser.parse_args()
	df_=pd.read_csv(args.input_file,sep='|')
	df_.columns=['utterance','text']
	# sentences_list=list(set(data['utterance'].to_list()))


	# from sklearn.model_selection import train_test_split
# 
	# sentences=[]

	# for sent in sentences_list:
	# 	subdata=data[data['utterance']==sent]
	# 	sentences.append([( getattr(row, "token"),getattr(row, "pos"),getattr(row, "lemma"),getattr(row, "utterance")) for index, row in subdata.iterrows() if not getattr(row, "token") in string.punctuation])

	# X = [sent2features(s) for s in sentences]
	
	# data=pd.DataFrame(flatten(X))
	# target_colomn=data.columns.to_list()

	# y = [sent2labels(s) for s in sentences]
	# utts=pd.DataFrame(flatten(y),columns=['utterance'])
	
	# df=pd.concat([data,utts], axis=1)
	
	# utterance_size_group=df.groupby('utterance').size()


	# NB_MAX_WORD=max(utterance_size_group)
	# NB_FEAT_WORD=df.shape[1]
	
	# columns=[ '{}_{}'.format(nvar,str(i).zfill(3))  for i in range(NB_MAX_WORD) for nvar in target_colomn ]
	# index=list(set(df['utterance'].to_list()))
	# df_ = pd.DataFrame(index=index, columns=columns)
	# # df_ = df_.fillna(0) # with 0s rather than NaNs
	# # data = np.array([np.arange(len(index))]*len(columns)).T
	# # df_ = pd.DataFrame(data, index=index, columns=columns)
	# # print(columns)
	# for  iutt,uttname in enumerate(index):
	# 	subdata=df[df.utterance==uttname]
	# 	subdata=subdata.drop(['utterance'], axis=1)
	# 	# print(subdata)
	# 	# print(subdata.keys().to_list())
	# 	utt=subdata.values.reshape((-1,len(subdata)*len(target_colomn)))
	# 	# print(utt.shape[1])
	# 	df_.at[uttname,:utt.shape[1]]=utt #subdata.values.reshape(-1,len(subdata)*NB_FEAT_WORD)
		
	# df_ = df_.fillna(0)
	# word_keys=['word.lower()_{}'.format(str(i).zfill(3)) for i in range(NB_MAX_WORD)]
	# lemma_keys=['lemma_{}'.format(str(i).zfill(3)) for i in range(NB_MAX_WORD)]
	# pos_keys=['postag_{}'.format(str(i).zfill(3)) for i in range(NB_MAX_WORD)]
	# # print(word_keys)
	# for idx in range(len(df_)):
	# 	df_.ix[idx, 'text'] =' '.join([val for val in df_.iloc[idx][word_keys].to_list() if val !=0 ])
	# 	df_.ix[idx, 'text_lemmatized'] =' '.join([val for val in df_.iloc[idx][lemma_keys].to_list() if val !=0 ])


	# df_=text_emotion(df_, 'text_lemmatized')
	df_['text'].apply(nltk.word_tokenize).apply(cleanTxt)
	df_['word_count'] = df_['text'].apply(nltk.word_tokenize).apply(len)
	NB_MAX_WORD=max(df_['word_count'])+1
	print(NB_MAX_WORD)
	# emotions=['joy','fear','sadness','anger','surprise','disgust','negative','positive']

	# print(max(df_['word_count'].to_list()),NB_MAX_WORD)


	# for emotion in emotions:
	# 	df_[emotion] = df_[emotion] / df_['word_count']
	# # df_.set_index(['utterance'], inplace=True)
	# print('TFIDF calculate information ')

	tfidf_data=TFIDF(df_['text'],MAX_NB_WORDS=NB_MAX_WORD)
	word_tfidf_keys=['word_tfidf_{}'.format(str(i).zfill(3)) for i in range(NB_MAX_WORD)]
	df_tfidf=pd.DataFrame(data=tfidf_data,columns=word_tfidf_keys)

	# df_=pd.concat([df_,df_tfidf], axis=1)



	# # # print(df_.head())

	# # # outFilename=os.path.join('./output/',os.path.basename(args.input_file).split('.')[0]+'.csv')
	# # # df_.to_csv(outFilename,sep='|',index=False)


	# # # we don't need the lemma and word anymore so we will drop the corresponding columns
	# # drop_col=word_keys +lemma_keys+['text', 'text_lemmatized']+pos_keys#, 'id']

	# # dataset=df_.drop(drop_col, axis=1)
	

	# # X_cat = dataset.copy()
	# # X_cat = dataset.select_dtypes(include=['object'])
	# # X_enc = X_cat.copy()

	# # print(pos_keys)
	# # X_enc = pd.get_dummies(X_enc, columns=pos_keys)
	# # dataset = dataset.drop(pos_keys,axis=1)


	# # finalData = pd.concat([dataset,X_enc], axis=1)
	# # print(finalData.head())

	standardizer=StandardScaler()
	# with sentiment labeling
	X_standard=standardizer.fit_transform(df_tfidf)
	X_standard=np.nan_to_num(X_standard)

	# # X_without_sent=X_without_sent[~np.isnan(X_without_sent)]
	# X_standard=np.nan_to_num(X_standard)
	print(X_standard.shape)


	# #using pca as dimension reduction technique

	PCA_model = PCA(0.90, random_state=42)

	X_standard = PCA_model.fit_transform(X_standard)*(-1)
	print('explained_variance_ratio_')
	print(PCA_model.explained_variance_ratio_)
	print(X_standard.shape)
	
	# # print(X_standard.shape)
	# # using T-distributed Stochastic Neighbor Embedding (T-SNE)
	# # from sklearn.manifold import TSNE
	# # X_standard = TSNE(n_components=2).fit_transform(X_standard)

	# # from sklearn.decomposition import NMF
	# # X_standard = NMF(n_components=2).fit_transform(X_standard)

	# # from sklearn import random_projection

	# # X_standard = random_projection.GaussianRandomProjection(n_components=2).fit_transform(X_standard)


	# # # Single VD
	# # from numpy import array
	# # from sklearn.decomposition import TruncatedSVD


	# # # X_standard = TruncatedSVD(n_components=2).fit_transform(X_standard)
	# # # https://medium.com/district-data-labs/modern-methods-for-sentiment-analysis-694eaf725244  
	# # print(X_standard.shape)

	# # # Elbow method 


	# # plt.xlim([0, 30])
	# # plt.ylim([0, 1])
	
	# # # k means determine k
	# # distortions = []
	# # K = range(2,30)
	# # for k in K:
	# # 	kmeanModel = KMeans(n_clusters=k).fit(X_standard)
	# # 	kmeanModel.fit(X_standard)
	# # 	percentage=(sum(np.min(cdist(X_standard, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_standard.shape[0])*100
	# # 	distortions.append(np.log(percentage))




	# # # # Plot the elbow
	# # plt.plot(K, distortions, 'bx-')
	# # plt.xlabel('k')
	# # plt.ylabel('Distortion')
	# # plt.title('The Elbow Method showing the optimal k')
	# # plt.show()

	range_n_clusters =np.arange(2,10,+2)
	# print(df_.shape)
	X_labeled,hyper_parm_turning=call_silhout_(X_standard,df_,range_n_clusters)


	hyper_parm_turning=pd.DataFrame(hyper_parm_turning)
	# Sort the rows of dataframe by column 'Name'
	hyper_parm_turning = hyper_parm_turning.sort_values(by =['silhouette_avg','sample_dist_std'],ascending=False)
	 
	print("Contents of Sorted Dataframe based on a single column 'silhouette_avg' & 'sample_dist_std' : ")
	print(hyper_parm_turning)

	# # X_labeled['utterance']=list(X_labeled.index)
	# # cluster_labeling=X_labeled.groupby('cluster_labels_10').size()
	# # print(cluster_labeling.index)
	# # for group in cluster_labeling:
	# # 	print(group)








if __name__ == '__main__':
	main()
