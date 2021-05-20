# T-distributed Stochastic Neighbor Embedding (T-SNE)

import numpy as np
from sklearn.manifold import TSNE


def main():
	
	X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
	X_embedded = TSNE(n_components=2).fit_transform(X)
	X_embedded.shape



if __name__ == '__main__':
	main()