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

from sklearn.preprocessing import scale


from gensim.models.word2vec import Word2Vec




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
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
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

		# Compute the silhouette scores for each sample
		sample_silhouette_values = silhouette_samples(X, cluster_labels)

		y_lower = 10
		for i in range(n_clusters):
			# Aggregate the silhouette scores for samples belonging to
			# cluster i, and sort them
			ith_cluster_silhouette_values = \
				sample_silhouette_values[cluster_labels == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.nipy_spectral(float(i) / n_clusters)
			ax1.fill_betweenx(np.arange(y_lower, y_upper),
							  0, ith_cluster_silhouette_values,
							  facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		ax1.set_title("The silhouette plot for the various clusters.")
		ax1.set_xlabel("The silhouette coefficient values")
		ax1.set_ylabel("Cluster label")

		# The vertical line for average silhouette score of all the values
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax1.set_yticks([])  # Clear the yaxis labels / ticks
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

		# 2nd Plot showing the actual clusters formed
		colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
		ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
					c=colors, edgecolor='k')

		# Labeling the clusters
		centers = clusterer.cluster_centers_
		# Draw white circles at cluster centers
		ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
					c="white", alpha=1, s=200, edgecolor='k')

		for i, c in enumerate(centers):
			ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
						s=50, edgecolor='k')

		ax2.set_title("The visualization of the clustered data.")
		ax2.set_xlabel("Feature space for the 1st feature")
		ax2.set_ylabel("Feature space for the 2nd feature")

		plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
					  "with n_clusters = %d" % n_clusters),
					 fontsize=14, fontweight='bold')

	plt.show()
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
    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus

def buildWordVector(text,size,ab_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += ab_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec




def main():
	parser = argparse.ArgumentParser(description="")

	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")

	# Add arguments
	
	
	parser.add_argument("input_file", help="The input file to be projected")
	# parser.add_argument("out_path_file", help="The input file to be projected")
	args = parser.parse_args()
	data=pd.read_csv(args.input_file,sep='\t')
	sentences_list=list(set(data['utterance'].to_list()))


	from sklearn.model_selection import train_test_split

	sentences=[]

	for sent in sentences_list:
		subdata=data[data['utterance']==sent]
		sentences.append([( getattr(row, "token"),getattr(row, "pos"),getattr(row, "lemma"),getattr(row, "utterance")) for index, row in subdata.iterrows() if not getattr(row, "token") in string.punctuation ])

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



	X_clean=cleanText(df_['text'].to_list())

	n_dim = 100
	#Initialize model and build vocab
	print('Initialize model and build vocab')
	ab_w2v = Word2Vec(size=n_dim, window=10, min_count=5, workers=4,alpha=0.025, min_alpha=0.025, iter=20)
	ab_w2v.build_vocab(X_clean)	
	print('Train the model over train_reviews (this may take several minutes)')
	ab_w2v.train(X_clean, epochs=ab_w2v.iter, total_examples=ab_w2v.corpus_count)

	from sklearn.preprocessing import scale
	print('building an average word2vec for each utterance (sentences)')
	df_word2vec = np.concatenate([buildWordVector(z, n_dim,ab_w2v) for z in X_clean])
	utt2vec_columns=['word2vec_{}'.format(str(i).zfill(3)) for i in  range(df_word2vec.shape[1])]
	df_word2vec=pd.DataFrame(df_word2vec,columns=utt2vec_columns)
	df_word2vec['utterance']=list(df_.index)

	print(df_word2vec.head())
	df_=text_emotion(df_, 'text_lemmatized')
	df_['word_count'] = df_['text_lemmatized'].apply(nltk.word_tokenize).apply(len)
	emotions=['joy','fear','sadness','anger','surprise','disgust','negative','positive']

	for emotion in emotions:
		df_[emotion] = df_[emotion] / df_['word_count']


	df_word2vec=pd.concat([df_word2vec,df_[emotions]],axis=1)
	from sklearn.preprocessing import scale
	train_vecs = scale(df_word2vec)




	# create label for each row
	sentence_emotion_labeling=[]
	for i,row  in df_[emotions].iterrows():
		label='_'.join([ '{}{}'.format(emotion,str(round(getattr(row,emotion),2))) for emotion in emotions if getattr(row,emotion)>0])
		if label=='':
			label='unknown'
		sentence_emotion_labeling.append(label)

	sentence_emotion_labeling 
			


	df_['sentence_label']=sentence_emotion_labeling



	# #using pca as dimension reduction technique

	PCA_model = PCA(n_components=2, random_state=42)
	X_standard = PCA_model.fit_transform(df_word2vec)*(-1)

	


	range_n_clusters =np.arange(3,10,+1)
	print(df_.shape)
	X_labeled=call_silhout_(X_standard,df_,range_n_clusters)
	X_labeled['utterance']=index
	X_labeled['sentence_label']=sentence_emotion_labeling

	cluster_labeling=X_labeled[['utterance','sentence_label','cluster_labels_8']].groupby('cluster_labels_8')

	cluster=''
	
	outPutData=OrderedDict()

	for idx,group in cluster_labeling:
		
		if cluster!=group['cluster_labels_8'].to_list()[0] and group.shape[0]>80 :
			cluster=group['cluster_labels_8'].to_list()[0]
			print('the shape of the group {} cluster name {}'.format(group.shape,cluster))
			# print(group['utterance'].to_list())
			# with codecs.open('./Doc2Vec/cluster_{}_doc2vec_with_emolex.scp'.format(cluster),'w','utf-8') as cluster:
			for utt,label in zip(group['utterance'].to_list(),group['sentence_label'].to_list()):
				if not 'utterance' in outPutData.keys():
					outPutData['utterance']=[utt]
				else:
					outPutData['utterance'].append(utt)

				if not 'emotion_label' in outPutData.keys():
					outPutData['emotion_label']=[label]
				else:
					outPutData['emotion_label'].append(label)

				if not 'cluster' in outPutData.keys():
					outPutData['cluster']=[cluster]
				else:
					outPutData['cluster'].append(cluster)

	final_data=pd.DataFrame(outPutData)
	final_data['fold']=np.zeros((final_data.shape[0]))
	kf = KFold(n_splits=5,shuffle=True)
	for idx,(train_index, test_index) in enumerate(kf.split(final_data)):
		# print("TRAIN:", train_index, "TEST:", test_index)
		final_data.at[test_index,'fold']=idx
		# print(X[test_index])
	outFilename='./Doc2Vec/cluster_{}_word2vec_with_emolex.csv'.format(os.path.basename(args.input_file))
	final_data.to_csv(outFilename,index=False)




if __name__ == '__main__':
	main()
