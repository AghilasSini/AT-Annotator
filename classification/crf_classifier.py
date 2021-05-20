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

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# nltk.corpus.conll2002.fileids()


# train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
# test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


# print(train_sents[0])


def word2features(sent, i):
	word = sent[i][0]
	postag = sent[i][1]

	features = {
		'bias': 1.0,
		'word.lower()': word.lower(),
		'word[-3:]': word[-3:],
		'word[-2:]': word[-2:],
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),
		'postag': postag,
		'postag[:2]': postag[:2],
	}
	if i > 0:
		word1 = sent[i-1][0]
		postag1 = sent[i-1][1]
		features.update({
			'-1:word.lower()': word1.lower(),
			'-1:word.istitle()': word1.istitle(),
			'-1:word.isupper()': word1.isupper(),
			'-1:postag': postag1,
			'-1:postag[:2]': postag1[:2],
		})
	else:
		features['BOS'] = True


	if i < len(sent)-1:
		word1 = sent[i+1][0]
		postag1 = sent[i+1][1]
		features.update({
			'+1:word.lower()': word1.lower(),
			'+1:word.istitle()': word1.istitle(),
			'+1:word.isupper()': word1.isupper(),
			'+1:postag': postag1,
			'+1:postag[:2]': postag1[:2],
		})

	else:
		features['EOS'] = True

	
	return features


def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, postag, label in sent]


def sent2tokens(sent):
	return [token for token, postag, label in sent]

# def main():
# 	X_train = [sent2features(s) for s in train_sents]
# 	y_train = [sent2labels(s) for s in train_sents]



# 	X_test = [sent2features(s) for s in test_sents]
# 	y_test = [sent2labels(s) for s in test_sents]



# 	crf = sklearn_crfsuite.CRF(
# 		algorithm='lbfgs',
# 		c1=0.1,
# 		c2=0.1,
# 		max_iterations=100,
# 		all_possible_transitions=True
# 	)
# 	crf.fit(X_train, y_train)


# 	# Evaluation


# 	y_pred = crf.predict(X_test)
# 	print(metrics.flat_classification_report(
# 		y_test, y_pred,  digits=3
# 	))


# if __name__ == '__main__':
# 	main()

input_path='/home/aghilas/Workspace/Experiments/SynPaFlex-Code/ml_template/clustering/dataset/input/quotetagging_02546.csv'
data=pd.read_csv(input_path,sep='\t')
sentences_list=list(set(data['utterance'].to_list()))


from sklearn.model_selection import train_test_split

sentences=[]

for sent in sentences_list:
	subdata=data[data['utterance']==sent]
	sentences.append([( getattr(row, "token"),str(getattr(row, "pos")),getattr(row, "label")) for index, row in subdata.iterrows()])

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



crf = sklearn_crfsuite.CRF(
	algorithm='lbfgs',
	c1=0.1,
	c2=0.1,
	max_iterations=100,
	all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Evaluation


y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(
	y_test, y_pred,  digits=3
))


y_pred=y = label_binarize(flatten(y_pred), classes=[0, 1, 2])
y_test=label_binarize(flatten(y_test), classes=[0, 1, 2])
n_classes = y_test.shape[1]
# print(n_classes)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
