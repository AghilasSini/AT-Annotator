
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from gensim.test.utils import get_tmpfile
import nltk


LabeledSentence = gensim.models.doc2vec.LabeledSentence


import multiprocessing

cores = multiprocessing.cpu_count()


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



def tokenize_text(text):
    tokens = []
    # for sent in nltk.sent_tokenize(text,language='french'):
    for word in nltk.word_tokenize(text,language='french'):
        if len(word) < 2:
            continue
        tokens.append(word.lower())
    return tokens


def tagger(data,tag):
	train_tagged = data.apply( lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[tag]), axis=1)
	return train_tagged

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


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
	transcription_data_file=args.input_file
	df_=pd.read_csv(transcription_data_file,sep='|')
	df_.columns=['utterance','text']

	df_.index = range(df_.shape[0])


	print(df_.head())

	# df_['text']=df_['text'].apply(nltk.word_tokenize)
	print(df_.head())
	train_tagged = df_.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=r.utterance), axis=1)	# print(unsup_reviews.head())

	# # print(X_clean.shape)



	model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
	model_dbow.build_vocab(train_tagged)

	# %%time
	for epoch in range(30):
	    model_dbow.train(utils.shuffle(train_tagged), total_examples=len(train_tagged.values), epochs=1)
	    model_dbow.alpha -= 0.002
	    model_dbow.min_alpha = model_dbow.alpha


	n_dim=300

	model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)
	model_dmm.build_vocab(train_tagged)

	# %%time
	for epoch in range(30):
	    model_dmm.train(utils.shuffle(train_tagged), total_examples=len(train_tagged.values), epochs=1)
	    model_dmm.alpha -= 0.002
	    model_dmm.min_alpha = model_dmm.alpha


	model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
	model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

	from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
	new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
	#Get training set vectors from our models
	x_doc2vec=OrderedDict()
	for utt,text in zip(df_['utterance'].to_list(),df_['text'].to_list()):
		tokens = model_dm.infer_vector(text)
		x_doc2vec[utt]=tokens

	df_doc2vec=pd.DataFrame(x_doc2vec).T
	df_doc2vec.columns=[ 'doc2vec_{}'.format(str(i).zfill(3)) for i in range(n_dim)]
	df_doc2vec['utterance']=df_doc2vec.index
	df_doc2vec.to_csv('output_doc2vec_features.csv',index=False)
	fname = get_tmpfile("my_doc2vec_model")
	model.save(fname)
	model = Doc2Vec.load(fname)  # you can continue training with the loaded model!


if __name__ == '__main__':
	main()