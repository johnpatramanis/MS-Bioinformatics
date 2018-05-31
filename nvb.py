import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sklearn
import scipy
from sklearn.decomposition import PCA
from pandas import DataFrame
from sklearn.datasets import make_multilabel_classification


def postprob(x1,m,s):
    if s==0:
        s=0.000001
    posteriorprob=(1/math.sqrt(2*math.pi*s))*math.exp((-(x1-m)**2)/(2*s))
    return posteriorprob


class nvb:
    def __init__(self,training,traininglabels,test):
        self.TR=training
        self.TEST=test
        self.TRL=traininglabels
        self.n=training.shape[0]
        self.nfeatures=training.shape[1]
        self.nclasses=traininglabels.shape[1]
    def prior(self):
        priors=[]
        for x in range(0,self.nclasses):
            apriori=0
            for y in self.TRL:
                if y[x]==1:
                    apriori+=1
            apriori=apriori/len(self.TRL)
            priors.append(apriori)
        self.priors=priors        
        return priors
    def getlabels(self):
        classmetrics=[]
        for C in range(0,self.nclasses):
            meanandsigmas=[]
            for x in range(0,self.nfeatures):
                mymean=np.mean([ self.TR[y][x] for y in range(0,len(self.TR)) if self.TRL[y][C]==1 ])
                myvar=np.var([ self.TR[y][x] for y in range(0,len(self.TR)) if self.TRL[y][C]==1 ])
                mynonmean=np.mean([ self.TR[y][x] for y in range(0,len(self.TR)) if self.TRL[y][C]==0 ])
                mynonvar=np.var([ self.TR[y][x] for y in range(0,len(self.TR)) if self.TRL[y][C]==0 ])
                meanandsigmas.append([mymean,myvar,mynonmean,mynonvar])
            classmetrics.append(meanandsigmas)
        #print(len(classmetrics),len(classmetrics[0]),len(classmetrics[0][0])) #for each class, for each feature, mean and var
        
        labels=[]
        for x in range(0,len(self.TEST)): #gia kathe point
            label=[]
            for y in range(0,self.nclasses): #classmetrics[y]=kathe classh periexei 20 listes
                probyes=np.prod([ postprob(self.TEST[x][j],classmetrics[y][j][0],classmetrics[y][j][1]) for j in range(0,self.nfeatures)])
                yes=self.priors[y]*probyes
                probno=np.prod([ postprob(self.TEST[x][j],classmetrics[y][j][2],classmetrics[y][j][3]) for j in range(0,self.nfeatures)])
                no=(1-self.priors[y])*probno
                if yes>=no:
                    label.append(1)
                if yes<no:
                    label.append(0)
            labels.append(label)
        self.labels=labels        
        return labels

    def getprobs(self): #bonus function!
        classmetrics=[]
        for C in range(0,self.nclasses):
            meanandsigmas=[]
            for x in range(0,self.nfeatures):
                mymean=np.mean([ self.TR[y][x] for y in range(0,len(self.TR)) if self.TRL[y][C]==1 ])
                myvar=np.var([ self.TR[y][x] for y in range(0,len(self.TR)) if self.TRL[y][C]==1 ])
                mynonmean=np.mean([ self.TR[y][x] for y in range(0,len(self.TR)) if self.TRL[y][C]==0 ])
                mynonvar=np.var([ self.TR[y][x] for y in range(0,len(self.TR)) if self.TRL[y][C]==0 ])
                meanandsigmas.append([mymean,myvar,mynonmean,mynonvar])
            classmetrics.append(meanandsigmas)
        
        labels=[]
        for x in range(0,len(self.TEST)): #gia kathe point
            label=[]
            for y in range(0,self.nclasses): #classmetrics[y]=kathe classh periexei 20 listes
                probyes=np.prod([ postprob(self.TEST[x][j],classmetrics[y][j][0],classmetrics[y][j][1]) for j in range(0,self.nfeatures)])
                yes=self.priors[y]*probyes
                probno=np.prod([ postprob(self.TEST[x][j],classmetrics[y][j][2],classmetrics[y][j][3]) for j in range(0,self.nfeatures)])
                no=(1-self.priors[y])*probno
                label.append(yes/(yes+no))
            labels.append(label)
        self.probs=labels        
        return labels   #gia kathe sample, gia kathe class episterfei pithanotita gia yes 
        
        
        
        
        
        
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

MYERRORS=[]
for wow in range(0,100):           
    X,Y = make_multilabel_classification(n_samples=200, n_features=20,n_classes=4,
    allow_unlabeled=False)
    y=nvb(X[0:190],Y[0:190],X[191:199])
    y.prior()
    y.getlabels()
    y.getprobs()
    MYERRORS.append(y.error(Y[191:199]))
print(np.mean(MYERRORS))

MYERRORS=[]
for i in range(0,10):
    indexestoremove=[ o for o in range(i*20,(i+1)*20)]
    test=X[i*20:(i+1)*20]
    testlabels=Y[i*20:(i+1)*20]
    training=np.delete(X,indexestoremove,axis=0)
    traininglabels=np.delete(Y,indexestoremove,axis=0)
    y=nvb(training,traininglabels,test)
    y.prior()
    y.getlabels()
    MYERRORS.append(y.error(testlabels))
print(np.mean(MYERRORS)) #Cross validation ( 10 fold )
