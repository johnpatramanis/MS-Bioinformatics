import numpy as np
import random
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from pandas import DataFrame



x,y = make_circles(n_samples=1000, factor=.6, noise=.1)
X1=x.T # D x N diastaseis (2,1000)
#print(X1)
X1mean=np.mean(X1, axis=1)
#print(X1mean.shape)#(2,1)
for x in range(0,len(X1.T)):
	X1.T[x]=X1.T[x]-X1mean
#print(X1)# new X1

answer = str(input("Would you like a random sigma? (Y/N)   "))
if answer=='Y':
	sigma=float(random.random())
	print('this is your sigma {} '.format(sigma))
if answer=='N':
	sigma=float(input('Please enter a sigma between 0 and 1   '))



D=X1.shape[0]
N=X1.shape[1]
K=2#dimention pou thelw
Wpoints=[random.uniform(-1,1) for x in range(0,D*K)]
W=np.array(Wpoints).reshape(D,K) #D*K
#print(W.T.shape) # K*D



M=np.matmul(W.T,W)+sigma*np.eye(K,K) #K*K

m=np.mean(X1, axis=1) #το μ ,D*1
#print(m.shape)

Elist=[0]
counter=0
if sigma!=0:
	while Elist[counter]<=0: 
		Eznlist=[]
		Eznzntlist=[]
		for column in X1.T:
			Ezn=np.matmul(np.matmul(np.linalg.inv(M),W.T),column-m)
			#print(Ezn.shape) #K*1, N fores
			Eznlist.append(Ezn)
			Eznznt=(sigma*np.linalg.inv(M))+np.matmul(Ezn,Ezn.T)
			#print(Eznznt.shape) #K*K diastaseis , Nfores
			Eznzntlist.append(Eznznt)
		E=-sum([((D/2)*np.log(2*np.pi*sigma))+((1/2)*np.trace(Eznzntlist[x]))+((1/(2*sigma))*np.linalg.norm(X1.T[x]-m)**2)-np.matmul(np.matmul((1/sigma)*Eznlist[x].T,W.T),(X1.T[x]-m))+(1/(2*sigma))*np.trace(np.matmul(np.matmul(Eznzntlist[x],W.T),W)) for x in range(0,len(Eznzntlist))])
		print(E)
		counter+=1
		Elist.append(E)
		W=(sum([np.matmul((X1[:,x]-m).reshape(D,1),Eznlist[x].reshape(D,1).T) for x in range(0,len(X1.T))]))* (np.linalg.inv(sum(Eznzntlist)))
		sigma=(1/(N*D))*sum([((np.linalg.norm(X1.T[x]-m))**2)-(2*(np.matmul(np.matmul(Eznlist[x].T,W.T),X1.T[x]-m)))+(np.trace(np.matmul(np.matmul(Eznzntlist[x],W.T),W))) for x in range(0,len(X1.T))])
		print(sigma)
		if abs(Elist[counter]-Elist[counter-1])<=0.00001:
			print('CONVERGENECE ACHIEVED!')
			break
X2=np.matmul(W.T,X1)
print('we\'ve reached {} dimensions'.format(X2.shape[0]))







if sigma==0:
	Wlist=[W]
	while len(Wlist)<=10000:
		OMEGA=np.matmul(np.matmul(np.linalg.inv(np.matmul(W.T,W)),W.T),X1)
		#print(OMEGA.shape)#2,1000 M,N
		W=np.matmul(X1,np.matmul(OMEGA.T,np.linalg.inv(np.matmul(OMEGA,OMEGA.T))))
		#print(W.shape)#2,2 D,M
		Wlist.append(W)
		counter+=1
		if np.linalg.norm(Wlist[counter]-Wlist[counter-1])==0:
			break
print(W)
X2=np.matmul(W.T,X1)
print('we\'ve reached {} dimensions'.format(X2.shape[0]))


	
	
	
	

df = DataFrame(dict(x=X2[0,:], y=X2[1,:], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

