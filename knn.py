import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
import scipy
from sklearn.decomposition import PCA
from pandas import DataFrame
from sklearn.datasets import make_multilabel_classification




def most_common(lst):
    return max(set(lst), key=lst.count)


class kNN:
	def __init__(self,training,traininglabels,test,k,metric):
		self.TR=training
		self.TEST=test
		self.k=k
		self.metric=metric
		self.TRL=traininglabels
		self.nclasses=traininglabels.shape[1]
	def getlabels(self):
		labels=[]
		for x in range(0,len(self.TEST)):
			distances=[]
			for y in range(0,len(self.TR)):
				if self.metric=='euclidean':
					distances.append([np.linalg.norm(self.TR[y]-self.TEST[x])**2,list(self.TRL[y])])
				if self.metric=='minkowski':
					distances.append([scipy.spatial.distance.minkowski(self.TR[y],self.TEST[x],p=2),list(self.TRL[y])])
				if self.metric=='manhattan':
					distances.append([scipy.spatial.distance.cityblock(self.TR[y],self.TEST[x]),list(self.TRL[y])])
				if self.metric=='cosine':
					distances.append([(1 - scipy.spatial.distance.cosine(list(self.TR[y]), list(self.TEST[x]))), list(self.TRL[y])])
				
			myks=sorted(distances)[0:self.k]
			mylabels=[]
			for j in range(0,self.nclasses):
				counts=[]
				for k in range(0,len(myks)):
					counts.append(myks[k][1][j])
				mylabels.append(most_common(counts))
			labels.append(mylabels)
		self.labels=labels
		return labels
	def error(self,truelabels):
		if len(self.labels)!=len(truelabels):
			print('error labels no the same length as test ')
		else:
			totalscore=0
			for x in range(0,len(self.labels)):
				for y in range(0,self.nclasses):
					if self.labels[x][y]==truelabels[x][y]:
						totalscore+=1/self.nclasses
			return totalscore/len(truelabels)

## lets do some tests!

meanerror=[]            
for R in range(0,10):				
	X,Y = make_multilabel_classification(n_samples=200, n_features=20,n_classes=4,
	allow_unlabeled=False)

		
	y=kNN(X[0:180],Y[0:180],X[181:199],3,'euclidean')
	y.getlabels()
	meanerror.append(y.error(Y[181:199]))
print(np.mean(meanerror)) #mean error for these settings

#lets try some different metrics
methods=['euclidean','minkowski','manhattan','cosine']
for M in methods:
    meanerror=[]            
    for R in range(0,10):				
        X,Y = make_multilabel_classification(n_samples=200, n_features=20,n_classes=4,
        allow_unlabeled=False)

            
        y=kNN(X[0:180],Y[0:180],X[181:199],3,M)
        y.getlabels()
        meanerror.append(y.error(Y[181:199]))
    print(np.mean(meanerror),M) #mean error,method
    
#lets test for the best k!, 10 fold valiation
X,Y = make_multilabel_classification(n_samples=200, n_features=20,n_classes=4,
allow_unlabeled=False)
Mykappas=[3,4,7,9,15,21,31,51,101,201]
tenfolderror=[]
Myerrors=[]
for k in Mykappas:				
	for i in range(0,10):
		indexestoremove=[ o for o in range(i*20,(i+1)*20)]
		print(indexestoremove)
		test=X[i*20:(i+1)*20]
		testlabels=Y[i*20:(i+1)*20]
		training=np.delete(X,indexestoremove,axis=0)
		traininglabels=np.delete(Y,indexestoremove,axis=0)
		print(i*20,(i+1)*20)
		y=kNN(training,traininglabels,test,k,'euclidean')
		y.getlabels()
		tenfolderror.append(y.error(testlabels))
	Myerrors.append(np.mean(tenfolderror))


df = DataFrame(dict(x=Mykappas, y=Myerrors))
fig, ax = plt.subplots()

df.plot(ax=ax, kind='scatter', x='x', y='y')
ax.set_xlabel("Number of Kappas ")
ax.set_ylabel("Success rate ")
plt.show()