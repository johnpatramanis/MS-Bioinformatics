import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import os
import time


os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS5nnn/GDS5430/soft/GDS5430.soft.gz')
os.system('gunzip GDS5430.soft.gz')
os.system("grep -v '^!' GDS5430.soft | grep -v '^^' | grep -v '^#' >GDS5430.soft.clean")

myfile='GDS5430.soft.clean'
MYDATA = pd.read_csv('GDS5430.soft.clean', sep='\t')
X1 = MYDATA[MYDATA.columns.difference(['ID_REF', 'IDENTIFIER'])]
X1=X1.values
print(X1.shape)

SCS1=[]
times1=[]
for K in range(2,9):
	start = time.time()
	pca = PCA(n_components=2)
	pca.fit(X1.T)
	X2=pca.transform(X1.T)
	wow=KMeans(K).fit(X2)
	end = time.time()
	labels=wow.labels_
	SCS1.append(sklearn.metrics.silhouette_score(X2,labels))
	times1.append(end - start)

	df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=labels))
	colors = {1:'red', 0:'blue',2:'green',3:'yellow',4:'c',5:'m',6:'k',7:'w'}
	fig, ax = plt.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
		group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
		
		
plt.show()
print(times1)
print(SCS1) #K=6 best Kappa


SCS2=[]
times2=[]
for K in range(2,9):
	start = time.time()
	pca = PCA(n_components=2)
	pca.fit(X1.T)
	X2=pca.transform(X1.T)
	wowzers=GaussianMixture(K).fit(X2).predict(X2)
	end = time.time()
	labels=wowzers
	SCS2.append(sklearn.metrics.silhouette_score(X2,labels))
	times2.append(end - start)

	
	df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=labels))
	colors = {1:'red', 0:'blue',2:'green',3:'yellow',4:'c',5:'m',6:'k',7:'w'}
	fig, ax = plt.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
		group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
		
plt.show()
print(times2)
print(SCS2) #K=5 best Kappa


####lets compar the 2 methods

Mytimes=times1+times2
MySCS=SCS1+SCS2
Mylabels=['kmeans' for x in range(0,7)]+['mixture' for x in range(0,7)]
Mykappas=[x for x in range(2,9)]+[x for x in range(2,9)]

df = DataFrame(dict(x=Mykappas, y=MySCS, label=Mylabels))
colors = {'kmeans':'red', 'mixture':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
	group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Coefficient")
plt.show()


df = DataFrame(dict(x=Mytimes, y=MySCS, label=Mylabels))
colors = {'kmeans':'red', 'mixture':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
	group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Silhouette Coefficient")
plt.show()

