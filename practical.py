import numpy as np
import sklearn
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import os


mean=[1 for x in range(0,10)]
cov=np.eye(10,10)
Data1=np.random.multivariate_normal(mean,cov,10)

print(Data1.shape)
COV=np.cov(Data1) # 2 x 2
EIGENVAL,EIGENVECT=np.linalg.eig(COV)
print(sorted(EIGENVAL))

COV=np.cov(Data1[0:4]) # 2 x 2
EIGENVAL,EIGENVECT=np.linalg.eig(COV)
print(sorted(EIGENVAL))









os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS5nnn/GDS5430/soft/GDS5430.soft.gz')
os.system('gunzip GDS5430.soft.gz')
os.system("grep -v '^!' GDS5430.soft | grep -v '^^' | grep -v '^#' >GDS5430.soft.clean")

myfile='GDS5430.soft.clean'
MYDATA = pd.read_csv('GDS5430.soft.clean', sep='\t')
X1 = MYDATA[MYDATA.columns.difference(['ID_REF', 'IDENTIFIER'])]
X1=X1.values
print(X1.shape)


SAMPLEID=open('sampleid.txt','r')
IDS=[]
for f in SAMPLEID:
	IDS.append(f.strip().split('\t'))
ALCOHOLISM=[x[2] for x in IDS[1:]]

def setlabels(list):
	labels=[]
	for x in list:
		if x[2]=='alcoholism' and x[3]=='male':
			labels.append('alcohol_male')
		if x[2]=='alcoholism' and x[3]=='female':
			labels.append('alcohol_female')
		if x[2]=='control' and x[3]=='male':
			labels.append('control_male')
		if x[2]=='control' and x[3]=='female':
			labels.append('control_female')
	return labels

################################################## alcohol/control + male/female Labels ###############################################




########################################################
pca = PCA(n_components=2)
pca.fit(X1.T)
X2=pca.transform(X1.T)
LABELS=setlabels(IDS[1:])

#df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=LABELS))
#colors = {'alcohol_male':'red', 'control_male':'blue','alcohol_female':'green','control_female':'yellow'}
#fig, ax = plt.subplots()
#grouped = df.groupby('label')
#for key, group in grouped:
#    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()

##################################################################
pcakernel=sklearn.decomposition.KernelPCA(n_components=2)
pcakernel.fit(X1.T)
X3=pcakernel.transform(X1.T)

#df = DataFrame(dict(x=X3[:,0], y=X3[:,1], label=LABELS))
#colors = {'alcohol_male':'red', 'control_male':'blue','alcohol_female':'green','control_female':'yellow'}
#fig, ax = plt.subplots()
#grouped = df.groupby('label')
#for key, group in grouped:
#    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()

########################################################################
#sparcepca=sklearn.decomposition.SparsePCA(n_components=2)
#sparcepca.fit(X1.T)
#X4=sparcepca.transform(X1.T)

#df = DataFrame(dict(x=X4[:,0], y=X4[:,1], label=LABELS))
#colors = {'alcohol_male':'red', 'control_male':'blue','alcohol_female':'green','control_female':'yellow'}
#fig, ax = plt.subplots()
#grouped = df.groupby('label')
#for key, group in grouped:
#    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()

###############################################################################

ppca=sklearn.decomposition.FactorAnalysis(n_components=2)
ppca.fit(X1.T)
X5=ppca.transform(X1.T)

#df = DataFrame(dict(x=X5[:,0], y=X5[:,1], label=LABELS))
#colors = {'alcohol_male':'red', 'control_male':'blue','alcohol_female':'green','control_female':'yellow'}
#fig, ax = plt.subplots()
#grouped = df.groupby('label')
#for key, group in grouped:
#    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()

################################################################### ethanol - alcoholic Labels ######################################


def setehtanlolabels(list):
	labels=[]
	for x in list:
		if x[1]=='ethanol':
			labels.append('ethanol')
		if x[1]=='untreated':
			labels.append('untreated')
	return labels


pca = PCA(n_components=2)
pca.fit(X1.T)
X2=pca.transform(X1.T)
labels=setehtanlolabels(IDS[1:])

df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=labels))
colors = {'ethanol':'red', 'untreated':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()


pcakernel=sklearn.decomposition.KernelPCA(n_components=2)
pcakernel.fit(X1.T)
X3=pcakernel.transform(X1.T)

df = DataFrame(dict(x=X3[:,0], y=X3[:,1], label=labels))
colors = {'ethanol':'red', 'untreated':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()

ppca=sklearn.decomposition.FactorAnalysis(n_components=2)
ppca.fit(X1.T)
X5=ppca.transform(X1.T)

df = DataFrame(dict(x=X5[:,0], y=X5[:,1], label=labels))
colors = {'ethanol':'red', 'untreated':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()


#sparcepca=sklearn.decomposition.SparsePCA(n_components=2)
#sparcepca.fit(X1.T)
#X4=sparcepca.transform(X1.T)

#df = DataFrame(dict(x=X4[:,0], y=X4[:,1], label=labels))
#colors = {'ethanol':'red', 'untreated':'blue'}
#fig, ax = plt.subplots()
#grouped = df.groupby('label')
#for key, group in grouped:
#    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()

###################################################### alcoholic/control Labels ##################################################

def settreatmentlabels(list):
	labels=[]
	samples=[]
	counter=0
	for x in list:
		if x[2]=='alcoholism':
			labels.append('alcoholic')
			samples.append(counter)
		if x[2]=='control':
			labels.append('control')
			samples.append(counter)
		counter+=1
	return labels,samples



labels=settreatmentlabels(IDS[1:])[0]
samples=settreatmentlabels(IDS[1:])[1]
print(samples)
X2=[]
for y in samples:
	X2.append(X1[y])
X2=np.asarray(X2)
print(X2)
#######################################################################
pca = PCA(n_components=2)
pca.fit(X2.T)
X3=pca.transform(X2.T)


df = DataFrame(dict(x=X3[:,0], y=X3[:,1], label=labels))
colors = {'alcoholic':'red', 'control':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

########################################################################
pcakernel=sklearn.decomposition.KernelPCA(n_components=2)
pcakernel.fit(X2.T)
X3=pcakernel.transform(X2.T)

df = DataFrame(dict(x=X3[:,0], y=X3[:,1], label=labels))
colors = {'alcoholic':'red', 'control':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()
