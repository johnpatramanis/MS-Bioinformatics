import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random
import math
import matplotlib.pyplot as plt
import sklearn
import scipy
from sklearn.decomposition import PCA
from pandas import DataFrame
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns




### first dataset
file=open('wdbc.data','r') #load it

labels=[]
data=[]
for l in file:
    y=l.strip().split(',') #set labels/datapoints
    labels.append(y[1])
    data.append(y[2:])
for x in range(0,len(data)):
    for y in range(0,len(data[x])):
        data[x][y]=float(data[x][y])

########################################################################
pca = PCA(n_components=2) #lets take a first look at our data
pca.fit(data)
X2=pca.transform(data)


df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=labels))
colors = {'B':'red', 'M':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()


########################################################################## 
#prepairing our dataset, choosing randomly 10% for testing and the rest for training
test=[]
testlabels=[]
indices=[x for x in range(0,len(data))]
usedindices=[]
for j in range(0,56):
    y=random.choice([x for x in indices if x not in usedindices]) #randomly choose around 10% of our data for test, the rest is used for training
    test.append(data[y])
    testlabels.append(labels[y])
    usedindices.append(y)
training=[]
traininglabels=[]
for x in indices:
    if x not in usedindices:
        training.append(data[x])
        traininglabels.append(labels[x])    

####################################################################
for x in range(0,len(test)): #turn our list into lists of arrays
    test[x]=np.asarray(test[x])

for x in range(0,len(training)):
    training[x]=np.asarray(training[x])

####################################################################
#now lets perform a knn classifications    
    
    
nbrs = KNeighborsClassifier(n_neighbors=3)
nbrs.fit(training, traininglabels)
wow=nbrs.predict(test)

success=0
for x in range(0,len(test)):
    print(wow[x],testlabels[x])
    if wow[x]==testlabels[x]:
        success+=1
print(success/len(test))#good success

############################################## lets find the best K!
k_list=[3,5,9,15,21,32,51,101]
cv_scores=[]
for k in k_list:
    nbrs = KNeighborsClassifier(n_neighbors=k)
    nbrs.fit(training, traininglabels)
    wow=nbrs.predict(test)

    scores = cross_val_score(nbrs, training, traininglabels, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
MSE = [1 - x for x in cv_scores]



plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.xticks(range(1,50,2))
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)

plt.show()


best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)


##################################################################
#and a random forest classification
wow=0
myforest=RandomForestClassifier(max_features=len(data[0]))
myforest.fit(training,traininglabels)
wow=myforest.predict(test)
success=0
for x in range(0,len(test)):
    print(wow[x],testlabels[x])
    if wow[x]==testlabels[x]:
        success+=1
print(success/len(test)) #extreme success!








############################################ SECOND DATASET ########################################################
############################################
#lets load the second dataset

R=open('breast-cancer-wisconsin.data')

labels=[]
data=[]
usedindices=[]
for l in R:
    y=l.strip().split(',') #set labels/datapoints
    labels.append(y[-1])
    data.append(y[1:])
for x in range(0,len(data)):
    for y in range(0,len(data[x])):
        if data[x][y]!='?': 
            data[x][y]=int(data[x][y])
        else:
            usedindices.append(data.index(data[x])) #afairoume ta NaN apo to test mas



       
 
        
##########################################################            
#prepairing our dataset
test=[]
testlabels=[]
indices=[x for x in range(0,len(data))]

for j in range(0,69):
    y=random.choice([x for x in indices if x not in usedindices]) #randomly choose around 10% of our data for test, the rest is used for training
    test.append(data[y])
    testlabels.append(labels[y])
    usedindices.append(y)
training=[]
traininglabels=[]
for x in indices:
    if x not in usedindices:
        training.append(data[x])
        traininglabels.append(labels[x])    
##############################################################
for x in range(0,len(test)): #turn our list into lists of arrays
    test[x]=np.asarray(test[x])

for x in range(0,len(training)):
    training[x]=np.asarray(training[x])
##############################################################
#now tha we removed missing data ,lets take a look
pca = PCA(n_components=2)
pca.fit(training)
X2=pca.transform(training)



df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=traininglabels))
colors = {'2':'red', '4':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()         
    
    
############################################################## 
#perform a knn classification  
nbrs = KNeighborsClassifier(n_neighbors=3)
nbrs.fit(training, traininglabels)
wow=nbrs.predict(test)

success=0
for x in range(0,len(test)):
    print(wow[x],testlabels[x])
    if wow[x]==testlabels[x]:
        success+=1
print(success/len(test))

k_list=[3,5,9,15,21,32,51,101]
cv_scores=[]
for k in k_list:
    nbrs = KNeighborsClassifier(n_neighbors=k)
    nbrs.fit(training, traininglabels)
    wow=nbrs.predict(test)

    scores = cross_val_score(nbrs, training, traininglabels, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
MSE = [1 - x for x in cv_scores]



plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.xticks(range(1,50,2))
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)

plt.show()

# finding best k
best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)





###############################################################
#perform a random forest classification
wow=0
myforest=RandomForestClassifier(max_features=len(data[0]))
myforest.fit(training,traininglabels)
wow=myforest.predict(test)
success=0
for x in range(0,len(test)):
    print(wow[x],testlabels[x])
    if wow[x]==testlabels[x]:
        success+=1
print(success/len(test)) #extreme!


