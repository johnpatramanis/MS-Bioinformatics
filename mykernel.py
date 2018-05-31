import numpy as np
import random
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

x,y = make_circles(n_samples=1000, factor=.6, noise=3)
X1=x.T # D x N diastaseis (2,1000)
print(X1.shape)

#initial image
df = DataFrame(dict(x=X1[0,:], y=X1[1,:], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()








def Gaussian(matrix,beta,n_components):
	sq_dists = pdist(matrix, 'sqeuclidean')
	#print(len(sq_dists))
	mat_sq_dists = squareform(sq_dists)
	#print(mat_sq_dists.shape)
	K = exp(-beta * mat_sq_dists)
	#print(K.shape)
	N = K.shape[0]
	one_n = np.ones((N,N)) / N
	#print(one_n)
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
	#print(K.shape)
	eigvals, eigvecs = eigh(K)
	X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
	return X_pc
	
	
	
	
	
def Polynomial(X1,p,components):
	polylist=[]
	N=X1.shape[1]
	for x in X1.T:
		for y in X1.T:
			polylist.append(np.matmul(x,y.T))
	print(len(polylist))
	polylist=np.asarray(polylist)
	polylist=(1+polylist)**p
	K=polylist.reshape(N,N)
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
	eigvals, eigvecs = eigh(K)
	X2 = np.column_stack((eigvecs[:,-i] for i in range(1,components+1)))
	return X2

	
	
	
	
def Hyperbolic(X1,delta,components):
	tanhlist=[]
	N=X1.shape[1]
	for x in X1.T:
		for y in X1.T:
			tanhlist.append(np.matmul(x,y.T))
	tanhlist=np.asarray(tanhlist)
	tanhlist=tanhlist-delta
	tanhlist=np.tanh(tanhlist)
	K=tanhlist.reshape(N,N)
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
	eigvals, eigvecs = eigh(K)
	X2 = np.column_stack((eigvecs[:,-i] for i in range(1,components+1)))
	return X2
	
	
	
def KernelPCA(matrix,type,extra,components):
	if str(type)=='gaussian':
		newmatrix=Gaussian(matrix.T,extra,components)
		return newmatrix
	if str(type)=='polynomial':
		newmatrix=Polynomial(matrix,extra,components)
		return newmatrix
	if str(type)=='hyperbolic':
		newmatrix=Hyperbolic(matrix,extra,components)
		return newmatrix
#X2=KernelPCA(X1,'polynomial',2,10)
#print(X2.shape)

type=str(input("Please choose which kernel to perform (gaussian,polynomial,hyperbolic  "))
K=int(input('Choose the dimensions you want to move to'))
extracomponent=float(input('Choose your extra component ( beta,delta,p)  '))


X2=KernelPCA(X1,type,extracomponent,K)
print(X2.shape)







df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()




