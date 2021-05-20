from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import NMF




def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with", str(np.array(X_train).shape[1]), "features")
    return (X_train, X_test)



from sklearn.datasets import fetch_20newsgroups


def main():
	newsgroups_train = fetch_20newsgroups(subset='train')
	newsgroups_test = fetch_20newsgroups(subset='test')
	X_train = newsgroups_train.data
	X_test = newsgroups_test.data
	y_train = newsgroups_train.target
	y_test = newsgroups_test.target

	X_train,X_test = TFIDF(X_train,X_test)

	NMF_ = NMF(n_components=2000)
	X_train_new = NMF_.fit(X_train)
	X_train_new =  NMF_.transform(X_train)
	X_test_new = NMF_.transform(X_test)

	print("train with old features: ",np.array(X_train).shape)
	print("train with new features:" ,np.array(X_train_new).shape)


	print("train with old features: ",np.array(X_train).shape)
	print("train with new features:" ,np.array(X_train_new).shape)


if __name__ == '__main__':
	main()