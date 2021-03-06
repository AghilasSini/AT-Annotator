from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec



x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)


#Do some very minor text preprocessing
def cleanText(corpus):
    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus


x_train = cleanText(x_train)
x_test = cleanText(x_test)


n_dim = 300
#Initialize model and build vocab
imdb_w2v = Word2Vec(size=n_dim, min_count=10)
imdb_w2v.build_vocab(x_train)


#Train the model over train_reviews (this may take several minutes)
imdb_w2v.train(x_train)


#Build word vector for training set by using the average value of all word vectors in the tweet, then scale

def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

from sklearn.preprocessing import scale


train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])


train_vecs = scale(train_vecs)

#Train word2vec on test tweets
imdb_w2v.train(x_test)


#Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])


test_vecs = scale(test_vecs)

