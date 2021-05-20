"""
===============================================================================
Selecting the number of clusters with silhouette analysis on KMeans clustering
===============================================================================

Silhouette analysis can be used to study the separation distance between the
resulting clusters. The silhouette plot displays a measure of how close each
point in one cluster is to points in the neighboring clusters and thus provides
a way to assess parameters like number of clusters visually. This measure has a
range of [-1, 1].

Silhouette coefficients (as these values are referred to as) near +1 indicate
that the sample is far away from the neighboring clusters. A value of 0
indicates that the sample is on or very close to the decision boundary between
two neighboring clusters and negative values indicate that those samples might
have been assigned to the wrong cluster.

In this example the silhouette analysis is used to choose an optimal value for
``n_clusters``. The silhouette plot shows that the ``n_clusters`` value of 3, 5
and 6 are a bad pick for the given data due to the presence of clusters with
below average silhouette scores and also due to wide fluctuations in the size
of the silhouette plots. Silhouette analysis is more ambivalent in deciding
between 2 and 4.

Also from the thickness of the silhouette plot the cluster size can be
visualized. The silhouette plot for cluster 0 when ``n_clusters`` is equal to
2, is bigger in size owing to the grouping of the 3 sub clusters into one big
cluster. However when the ``n_clusters`` is equal to 4, all the plots are more
or less of similar thickness and hence are of similar sizes as can be also
verified from the labelled scatter plot on the right.
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import confusion_matrix


from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


import plotnine as p9


print(__doc__)



# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
# X, y = make_blobs(n_samples=500,
#                   n_features=2,
#                   centers=4,
#                   cluster_std=1,
#                   center_box=(-10.0, 10.0),
#                   shuffle=True,
#                   random_state=1)  # For reproducibility


df=pd.read_csv('./dataset/espf-12clusters.csv',sep=',')

all_features=df.iloc[:,4:]
y=df.iloc[:,3]




#{('JOSPF', 12), ('FDB', 16), ('COSPF', 10), ('ANXIOUS', 1), ('ENCOURAGING', 5), 
#('ASSERTIVE', 3), ('WDB', 20), ('NEUTRAL', 8), ('NDB', 18), ('LDB', 17), ('ANGRY', 0), ('NESPF', 13), 
#('CONCERNED', 4), ('DESPF', 11), ('TDB', 19), ('APOLOGETIC', 2), ('ADB', 14), ('EDB', 15), ('SAD', 9), ('EXCITED', 6), ('HAPPY', 7)}



all_features_norm = StandardScaler().fit_transform(all_features)

PCA_model = PCA(n_components=2, random_state=42)
X = PCA_model.fit_transform(all_features_norm)*(-1)






# label_dict={ 1:'002 - DE', 5:'006 - TR' , 4:'005 - SU', 3:'004 - NE',2:'003 - JO',0 :'001 - CO'}
#label_dict={ 1:'002 - DE', 5:'006 - TR' , 4:'005 - SU', 3:'004 - NE',2:'003 - JO',0 :'001 - CO'}
label_dict={}
for lb,lbn in  zip(df.iloc[:,2].values,df.iloc[:,3].values):
	label="{} - {}".format(str(lbn+1).zfill(3),lb)
	if not lbn in label_dict.keys():
		label_dict[lbn]=label



# logging.info("Generate the PCA data frame for %s"  % input_file)
projected_df = pd.DataFrame(X)
projected_df.columns = ["Comp. 0", "Comp. 1"]




# # range_n_clusters =[12]
# # for n_clusters in range_n_clusters:

n_clusters=12
	
clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X)


centers = clusterer.cluster_centers_
x_centers=[]
y_centers=[]

labels_centers=[]
for ivalue,val in enumerate(centers):
	x_centers.append(val[0])
	y_centers.append(val[1])
	labels_centers.append(label_dict[ivalue])



projected_df['emotions']=[label_dict[lb] for lb in  cluster_labels]



# df['label_orginal']=[label_dict[yl] for yl in  y]




# # cm_=confusion_matrix(y,cluster_labels)
# #print('Confusion matrix with number of cluster {}'.format(n_clusters))
# #print(cm_)

# silhouette_avg = silhouette_score(X, cluster_labels)
# print("For n_clusters =", n_clusters,
# 	  "The average silhouette_score is :", silhouette_avg)

# # Compute the silhouette scores for each sample
# sample_silhouette_values = silhouette_samples(X, cluster_labels)


p = (
	p9.ggplot(projected_df, p9.aes(x="Comp. 0", y="Comp. 1", fill="emotions"))
	+ p9.geom_point()
	# + p9.geom_label(label = labels_centers)#p9.aes(y =y_centers, label = labels_centers))
	+ p9.theme(figure_size=(15, 15))
	+ p9.theme(legend_position=(1.05,0.45))
)
p.save("{}".format('output_plot_file.pdf'), dpi=400)


