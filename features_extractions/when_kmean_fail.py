import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

#Toy data sets
centers_neat = [(-10, 10), (0, -5), (10, 5)]
x_neat, _ = datasets.make_blobs(n_samples=5000, 
																centers=centers_neat,
																cluster_std=2,
																random_state=2)

x_messy, _ = datasets.make_classification(n_samples=5000,
																					n_features=10,
																					n_classes=3,
																					n_clusters_per_class=1,
																					class_sep=1.5,
																					shuffle=False,
																					random_state=301)
#Default plot params
plt.style.use('seaborn')
cmap = 'tab10'

# plt.figure(figsize=(17,8))
# plt.subplot(121, title='"Neat" Clusters')
# plt.scatter(x_neat[:,0], x_neat[:,1])
# plt.subplot(122, title='"Messy" Clusters')
# plt.scatter(x_messy[:,0], x_messy[:,1])
# plt.show()

# 
from sklearn.cluster import KMeans

#Predict K-Means cluster membership
km_neat = KMeans(n_clusters=3, random_state=2).fit_predict(x_neat)
km_messy = KMeans(n_clusters=3, random_state=2).fit_predict(x_messy)

# plt.figure(figsize=(15,8))
# plt.subplot(121, title='"Neat" K-Means')
# plt.scatter(x_neat[:,0], x_neat[:,1], c=km_neat, cmap=cmap)
# plt.subplot(122, title='"Messy" K-Means')
# plt.scatter(x_messy[:,0], x_messy[:,1], c=km_messy, cmap=cmap)
# plt.show()


from sklearn.mixture import GaussianMixture

#Predict GMM cluster membership
gm_messy = GaussianMixture(n_components=3).fit(x_messy).predict(x_messy)

# plt.figure(figsize=(15,8))
# plt.subplot(121, title='"Messy" K-Means')
# plt.scatter(x_messy[:,0], x_messy[:,1], c=km_messy, cmap=cmap)
# plt.subplot(122, title='"Messy" GMM')
# plt.scatter(x_messy[:,0], x_messy[:,1], c=gm_messy, cmap=cmap)
# plt.show()

import hdbscan


    #Toy data set
blob1, y1 = datasets.make_blobs(n_samples=25, 
                               centers=[(10,5)],
                               cluster_std=1.5,
                               random_state=2)

blob2, y2 = datasets.make_blobs(n_samples=500, 
                               centers=[(6,2)],
                               cluster_std=1.3,
                               random_state=2)

blob3, y3 = datasets.make_blobs(n_samples=500, 
                               centers=[(2,5)],
                               cluster_std=1,
                               random_state=2)

unbal = np.vstack([blob1, blob2, blob3])
y1[y1 == 0] = 0
y2[y2 == 0] = 1
y3[y3 == 0] = 2
labs = np.concatenate([y1, y2, y3])

#Predict K-Means cluster membership
km_unbal = KMeans(n_clusters=3, random_state=2).fit(unbal)
km_unbal_preds = KMeans(n_clusters=3, random_state=2).fit_predict(unbal)

plt.figure(figsize=(15,8))
plt.subplot(121, title='Generated Clusters and Assignments')
plt.scatter(unbal[:,0], unbal[:,1], c=labs, cmap=cmap)
plt.subplot(122, title='K-Means w/ Cluster Assignments and Centers')
plt.scatter(unbal[:,0], unbal[:,1], c=km_unbal_preds, cmap=cmap)
plt.scatter(km_unbal.cluster_centers_[:,0], km_unbal.cluster_centers_[:,1], marker='X', s=150, c='black')




clust_count = np.linspace(1, 20, num=20, dtype='int')

clust_number = 2
plot_number = 1
plt.figure (figsize=(17,12))
while clust_number < 21:
    hdb = hdbscan.HDBSCAN(min_cluster_size=clust_number)
    hdb_pred = hdb.fit(unbal)
    plt.subplot(5, 4, plot_number, title = 'Min. Cluster Size = {}'.format(clust_number))
    plt.scatter(unbal[:,0], unbal[:,1], c=hdb_pred.labels_, cmap=cmap)
    plot_number += 1
    clust_number += 1
    
plt.tight_layout()
plt.show()