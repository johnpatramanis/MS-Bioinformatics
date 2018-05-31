import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from pandas import DataFrame


mean=[2,2,1]
cov=0.25*np.eye(3,3)
Data1=np.random.multivariate_normal(mean,cov,220)
mean=[1,2,1]
cov=0.75*np.eye(3,3)
Data2=np.random.multivariate_normal(mean,cov,280)
Data1=Data1.T
Data2=Data2.T
Data=np.concatenate((Data1,Data2),axis=1)
print(Data.shape)
m=np.mean(Data, axis=1)

###############################
def SC(Data,Kpoints,Klabels):
	totalsum=[]
	for x in range(0,len(Kpoints)):
		Ksum=[]
		for y in range(0,len(Klabels)):
			if Klabels[y]==x:
				a=np.mean([ (np.linalg.norm(Data.T[y]-j))**2 for j in Data.T if (Data.T[y]!=j).all() ])
				b=min([(np.linalg.norm(Data.T[y]-j))**2 for j in Kpoints ])
				ab=[a,b]
				s=(a-b)/max(ab)
				Ksum.append(s)
		return np.mean(Ksum)	


def most_common(lst):
    return max(set(lst), key=lst.count)

#############################################################################




def Mixtureofgaussian(Data,K,stop,reps): #input
	repinfo=[]
	for w in range(0,reps):
		countner=0
		D,N = Data.shape
		Kpoints=[random.choice(Data.T) for x in range(0,K)] #K vectors in a list (means)
		beginingscore=sum([np.linalg.norm(x-y)**2 for x in Kpoints for y in Kpoints if (x!=y).all()])
		counter=0 #resets for each rep
		Sigmas=[np.eye(D) for x in range(0,K)] #list ofcovariance matrices (D,D)
		Mixcoeff= [ 1/K for x in range(0,K)]
		mylogs= []
		Gammas=[0 for x in range(0,K)] #gia na vrw to 1o log likelihood
		for x in range(0,K):
			Gammas[x]=[Mixcoeff[x]*((np.exp(np.matmul(-(1/2)*((y-Kpoints[x]).T),np.matmul(np.linalg.inv(Sigmas[x]),(y-Kpoints[x])))))/((2*np.pi)**(D/2))*(np.linalg.det(Sigmas[x]))**(1/2)) for y in Data.T] #lista me N stoixeia
			Gammas[x]=np.asarray(Gammas[x]) #N*1
		Gammamatrix=np.concatenate(Gammas).reshape(N,K) #(N,K)
		mylogs.append(np.sum(np.log(np.sum(Gammamatrix, axis = 1)))) #1o Log
		while counter<=stop:
			for x in range(0,K):
				Gammas[x]=[Mixcoeff[x]*((np.exp(np.matmul(-(1/2)*((y-Kpoints[x]).T),np.matmul(np.linalg.inv(Sigmas[x]),(y-Kpoints[x])))))/((2*np.pi)**(D/2))*(np.linalg.det(Sigmas[x]))**(1/2)) for y in Data.T] #lista me N stoixeia
				Gammas[x]=np.asarray(Gammas[x]) #N*1
			Gammamatrix=np.concatenate(Gammas).reshape(N,K) #(N,K)
			Gammamatrixorig=Gammamatrix
			for x in range(0,len(Gammamatrix)):
				Gammamatrix[x]=Gammamatrix[x]/sum(Gammamatrix[x]) # kathe grammh/sum ths
			for x in range(0,K):
				Nk=[]
				NkXn=[]
				for y in range(0,N):
					Nk.append(Gammamatrix[y][x]) #gia na vgaloume to sum ths kwlonas
					NkXn.append(Gammamatrix[y][x]*Data.T[y]) #
				Nk=sum(Nk)
				NkXn=sum(NkXn)
				Kpoints[x]=NkXn/Nk #new Kpoints (means)
				NkXnMk=[]
				for y in range(0,N):
					NkXnMk.append(Gammamatrix[y][x]*np.matmul(np.reshape((Data.T[y]-Kpoints[x]).T,(D,1)),np.reshape(Data.T[y]-Kpoints[x],(1,D))))
				NkXnMk=sum(NkXnMk)
				Sigmas[x]=NkXnMk/Nk #new sigma D,D
				Mixcoeff[x]=Nk/N #new mixing coef
			Loglikelihood=np.sum(np.log(np.sum(Gammamatrixorig, axis = 1)))
			print(Loglikelihood)
			mylogs.append(Loglikelihood)
			#print(Kpoints)
			#print(Sigmas)
			#print(Mixcoeff)
			counter+=1
			if counter>=2:
				if counter>=stop or (abs(mylogs[counter]-mylogs[counter-1])<=1e-16 and abs(mylogs[counter-1]-mylogs[counter-2])<=1e-16):
					print('achieved')
					break		
		Kassigned=[] #periexei enan arithmo (poi cluster anoikei) gia kathe shmeio
		for j in Gammamatrix:
			Kassigned.append(int(list(j).index(sorted(list(j))[-1])))
		Kclusters=[[] for x in range(0,K)] #periexei listes == arithmo cluster,kathe cluster poia shmeia periexei
		for x in range(0,N):
			Kclusters[Kassigned[x]].append(Data.T[x])
		Kclusters=[np.asarray(Kclusters[x]).T for x in range(0,K)]
		KLabels=[[] for x in range(0,K)] #lista me listes,1 gia kathe cluster periexoun ta index gia ta shmeia edos kathe cluster
		for x in range(0,len(KLabels)):
			for y in range(0,len(Kclusters[x].T)):
				ix=np.isin(Data.T,Kclusters[x].T[y])
				KLabels[x].append(np.where(ix)[0][0]) #to index tou shmeioy sto data mou
		
		
		Score=float(sum([(np.linalg.norm(x-y))**2 for j in range(0,K) for x in Kpoints[j] for y in Kclusters[j] ]))
		repinfo.append([Score,Kpoints,Kclusters,Kassigned])
	beginingscore=sorted(repinfo,key=lambda x: x[0])[0][0]
	Kpoints=sorted(repinfo,key=lambda x: x[0])[0][1]	
	Kclusters=sorted(repinfo,key=lambda x: x[0])[0][2]
	Kassigned=sorted(repinfo,key=lambda x: x[0])[0][3]
	return Kpoints,KLabels,Kassigned,beginingscore

wow=Mixtureofgaussian(Data,3,2000,1) #it does not converge!
print(wow[2])

clst1=most_common(wow[2][0:219])
corrects=[]
counter=1
for p in wow[2]:
	if counter<=220 and p==clst1:
		corrects.append('correct')
	if counter>220 and p!=clst1:
		corrects.append('correct')
	if (counter<=220 and p!=clst1) or (counter>220 and p==clst1):
		corrects.append('mistake')
	counter+=1

print(corrects.count('correct')) #235 (sto diko mou run), pretty bad

#as kanoume kai ena plot
pca = PCA(n_components=2)
pca.fit(Data.T)
X2=pca.transform(Data.T)
pca.fit(np.asarray(wow[0]))
X3=pca.transform(np.asarray(wow[0]))




df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=wow[2]))
colors = {0:'red', 1:'blue',2:'green',3:'yellow',4:'purple'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
for x in X3:
	ax.plot([x[0]],[x[1]],'kx', markersize=10)
plt.show()  #really bad 

