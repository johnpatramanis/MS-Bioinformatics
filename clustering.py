import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
import scipy
from sklearn.decomposition import PCA
from pandas import DataFrame
from scipy.spatial.distance import cdist




mean=[2,2,1]
cov=0.25*np.eye(3,3)
Data1=np.random.multivariate_normal(mean,cov,220)
mean=[1,2,1]
cov=0.75*np.eye(3,3)
Data2=np.random.multivariate_normal(mean,cov,280)
Data1=Data1.T
Data2=Data2.T
Data=np.concatenate((Data1,Data2),axis=1)
print(Data.shape) #(3,500)

Labels=['1' for x in range(0,220)]+['2' for x in range(0,280)]
print(len(Labels))

pca = PCA(n_components=2)
pca.fit(Data.T)
X2=pca.transform(Data.T)

df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=Labels))
colors = {'1':'red', '2':'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()  #we can see it is a bit messy

def most_common(lst):
    return max(set(lst), key=lst.count)


def SC(Data,Kpoints,Klabels):
	totalsum=[]
	for x in range(0,len(Kpoints)):
		Ksum=[]
		for y in range(0,len(Klabels)):
			if Klabels[y]==x:
				a=np.mean([ (np.linalg.norm(Data.T[y]-Data.T[j]))**2 for j in range(0,len(Data.T)) if (Data.T[y]!=Data.T[j]).all() and (Klabels[j]==x)])
				b=sorted([(np.linalg.norm(Data.T[y]-j))**2 for j in Kpoints ])[1]
				ab=[a,b]
				s=(a-b)/max(ab)
				Ksum.append(s)
		totalsum.append(np.mean(Ksum))
	return np.mean(totalsum)
	#thelw gia kathe shmeio tou data to s(i) opou s(i)=a(i)-b(i)/max(a:b) kai a(i)=mesos oros apostaseis apo alla points tou cluster kai b(i)=apostash apo kodinotero clusrter
	#prepei -1<s<1 kai na vgalw meso oro

	
	

def Kmeansclusttering(Data,K,stop,reps,type): #input
	repinfo=[]
	for w in range(0,reps):
		Kpoints=[random.choice(Data.T) for x in range(0,K)] #K vectors in a list
		counter=0 #resets for each rep
		score=[0] #resets for each rep
		while K>=0: 
			Kassigned=[[] for x in range(0,K)]
			Kclustered=[[] for x in range(0,K)]
			for x in Data.T:
				dist=[]
				for y in Kpoints:
					if str(type)=='euclidean':
						dist.append(np.linalg.norm(x-y)**2) #different methods of distance here
					if str(type)=='mahalanobis':
						dist.append(scipy.spatial.distance.mahalanobis(x,y,np.linalg.inv(np.cov(np.vstack((x,y)).T))))  #will not work if cov x,y not invertable
					if str(type)=='manhattan':
						dist.append(scipy.spatial.distance.cityblock(x,y))
				Kassigned[dist.index(min(dist))].append(x)
				Kclustered.append(dist.index(min(dist)))
			Kassigned=[np.asarray(Kassigned[x]).T for x in range(0,K)]
			Kpointsnew=[[] for x in range(0,K)]
			for x in range(0,K):
				if Kassigned[x]!=[]:
					Kpointsnew[x]=np.mean(Kassigned[x], axis=1) #new points	
				if Kassigned[x]==[]:
					Kpointsnew[x]=Kpoints[x]
			counter+=1
			score.append(0)
			for j in range(0,K):
				if (Kpointsnew[j]==Kpoints[j]).all(): #compare new points to old
					score[counter]+=1
			Kpoints=Kpointsnew # the old become the new
			print(score[counter]) #my current score
			if score[counter]==K: #otan score=K tote kathe palio point = kainourgio => Convergence
				break
				
			if counter>=stop: #user selected
				break
		beginingscore=float(sum([(np.linalg.norm(x-y))**2 for j in range(0,K) for x in Kpoints[j] for y in Kassigned[j] ]))
		repinfo.append([beginingscore,Kpoints,Kassigned,Kclustered])

	#Choose smallest sum
	beginingscore=sorted(repinfo,key=lambda x: x[0])[0][0] #to kalutero ,gia ama to theloume''score'' , apostasei edos kathe cluster
	Kpoints=sorted(repinfo,key=lambda x: x[0])[0][1] #ta kendra
	Kassigned=sorted(repinfo,key=lambda x: x[0])[0][2] #lista me listes, kathe mia einai ena cluster k periexei ta shmeia pou tou anoikoun
	Kclustered=sorted(repinfo,key=lambda x: x[0])[0][3][2:] #lista me N arithmous enas gia kathe shmeio,se poio cluster anoikei
	KLabels=[[] for x in range(0,K)] #lista me listes,kathe mia ena cluster, periexei ta index gia to poia shmeia einai sto kathe clsuter
	for x in range(0,len(KLabels)):
		for y in range(0,len(Kassigned[x].T)):
			ix=np.isin(Data.T,Kassigned[x].T[y])
			KLabels[x].append(np.where(ix)[0][0]) #np.where(jx)[0][0]) == index of Kassigned[x].T[y] in Data.T
	Kclustered=[x for x in Kclustered if x!=[] ]
	return Kpoints,KLabels,beginingscore,Kclustered # which cluster contains which sample

##################################################################################################################################################	
	
######### kseroume se poio cluster anoikoun kanonika ta shmeia mas, opote boroume na vathmologisoume to clsutering mas
wow=Kmeansclusttering(Data,2,100,5,'euclidean')
clst1=most_common(wow[3][0:219])
corrects=[]
counter=1
for p in wow[3]:
	if counter<=220 and p==clst1:
		corrects.append('correct')
	if counter>220 and p!=clst1:
		corrects.append('correct')
	if (counter<=220 and p!=clst1) or (counter>220 and p==clst1):
		corrects.append('mistake')
	counter+=1

print(corrects.count('correct')) #373 (sto diko mou run)

#as kanoume kai ena plot
wow=Kmeansclusttering(Data,2,100,5,'euclidean')
pca = PCA(n_components=2)
pca.fit(Data.T)
X2=pca.transform(Data.T)

pca.fit(np.asarray(wow[0]))
X3=pca.transform(np.asarray(wow[0]))
print(X3.shape)



df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=wow[3]))
colors = {0:'red', 1:'blue',2:'green',3:'yellow'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
for x in X3:
	ax.plot([x[0]],[x[1]],'kx', markersize=10)
plt.show()  #pretty nice
	
	
	
	
#####################################################################################################
	
mysilhouetes=[]
for KK in range(2,5):
	wow=Kmeansclusttering(Data,KK,100,5,'euclidean') #buggarei kamia fora,xanatrexoume 
	mysilhouetes.append(SC(Data,wow[0],wow[3]))
print(mysilhouetes) 

for K in range(2,5):
	wow=Kmeansclusttering(Data,K,100,5,'euclidean')
	pca = PCA(n_components=2)
	pca.fit(Data.T)
	X2=pca.transform(Data.T)
	pca.fit(np.asarray(wow[0]))
	X3=pca.transform(np.asarray(wow[0]))
	
	
	
	
	
	df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=wow[3]))
	colors = {0:'red', 1:'blue',2:'green',3:'yellow',4:'purple'}
	fig, ax = plt.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
		group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	for x in X3:
		ax.plot([x[0]],[x[1]],'kx', markersize=10)		
	plt.show()  #pretty nice
	
K=mysilhouetes.index(max(mysilhouetes))+2 #dialegoume to theoritika kalutero K
#ara
print(K)	



