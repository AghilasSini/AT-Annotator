#Word Embading
# coding: utf-8
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import argparse
import glob
import os
import sys
import re


def build_args():
	parser=argparse.ArgumentParser(description='')
	parser.add_argument('txtDir', type=str, nargs=1, help='data input')
	parser.add_argument('out_model', type=str, nargs=1, help='input')
	return parser.parse_args()





def extract_sentences(TextDir): 
    """
    Turns a collection of plain text files into a list of lists of word tokens.
    """
    print("--extract_sentences")
    Sentences = []
    for File in glob.glob(TextDir+'/*.txt'):
        with open(File, "r") as InFile: 
            Text = InFile.read()
            Text = re.sub("\n", " ", Text)
            Text = re.sub("--", "", Text)
            Text = re.sub("\.\.\.", ".", Text)
            Text = Text.lower()
            SentencesOne = []
            Text = re.split("[.!?]", Text)
            for Sent in Text: 
                Sent = re.split("\W", Sent)
                Sent = [Token for Token in Sent if Token]
                SentencesOne.append(Sent)  
            Sentences.extend(SentencesOne)
    return Sentences








# define training data
def load_all_sentences(all_normed_sentences,tokenizer):
	all_tokenized_sentences=[]
	for nsent in all_normed_sentences:
		all_tokenized_sentences.append(tokenizer.tokenize(nsent.decode('utf-8').lower()))
	return all_tokenized_sentences

# def get_slt_sentences():

def main():
	args=build_args()
	# tokenizer = nltk.RegexpTokenizer(r'\w+')
	# sentences=data.iloc[:,1].values

	# all_tokenized_sentences=load_all_sentences(sentences,tokenizer)
	all_tokenized_sentences=extract_sentences(args.txtDir[0])
	
	# 
	#train
	model = Word2Vec(all_tokenized_sentences, size=100, window=3, min_count=1, workers=4)
	words = list(model.wv.vocab)
	# with open('cmu_vocab.txt','w') as fout:
	# 	for wrd in model.wv.vocab:
	# 		fout.write(wrd+'\n')
	# model.save(args.out_model[0])
	new_model = Word2Vec.load(args.out_model[0])
	x=np.zeros((len(words),100))
	with open('cmu_wrd_emb.txt','w') as embf:
		for iwrd,wrd in enumerate(words):
			if wrd in list(new_model.wv.vocab):
				x[iwrd,:]=new_model[wrd]
			else:
				pass
	
		for iwrd,wrd in enumerate(words):
			embf.write("{} {}\n".format(wrd," ".join([str(feat) for feat in x[iwrd,:]])))
	# # for sent in sentences[:60]:
	# # 	print(sent,)
	# # 	cur_sent=tokenizer.tokenize(sent.lower())
	# # 	print(cur_sent)
	# # 	for word in cur_sent:
	# # 		print(new_model[word])

if __name__ == '__main__':
	main()


# # train model
# model = Word2Vec(sentences, min_count=1)
# # summarize the loaded model
# words = list(model.wv.vocab)
# # save model
# model.save('model.bin')
# # load model
# new_model = Word2Vec.load('model.bin')
# win_word =['je','azul','ici']
# win_array = np.zeros((100,99))


# for iword,word in enumerate(win_word):
#     print(new_model[word])
# #word_mat = win_word_em.reshape(100,3)
# # build up matrix


# # name of the file is associated to label of the current word
# import scipy.misc
# scipy.misc.imsave('outfile.jpg', win_array)
