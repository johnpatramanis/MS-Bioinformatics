import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from pandas import DataFrame

x,y = make_circles(n_samples=1000, factor=.6, noise=.1)
X1=x.T # D x N diastaseis (2,1000)
D,N=X1.shape
K=int(input("Please insert the number of dimensions you want to move to (must be <={} ) ".format(D)))
type=str(input('select method of decomposition ( SVD - default / LUL )  '))
print('our current dimensions are : {} '.format(X1.shape))
print('lets have a look')

#Mia prwth eikona
df = DataFrame(dict(x=X1[0,:], y=X1[1,:], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()


X1mean=np.mean(X1, axis=1)
for x in range(0,len(X1.T)):
	X1.T[x]=X1.T[x]-X1mean
COV=np.cov(X1) # 2 x 2
print(COV)

if type=='LUL':
	EIGENVAL,EIGENVECT=np.linalg.eig(COV)
	#print(EIGENVAL.shape) #oi 2 idiotimes
	#print(EIGENVECT.shape) # o U pinakas (eigenvectors) apo to LUL-1 decomposition
	X3=np.matmul(EIGENVECT.T[0:K],X1) # 2x2 * 2x1000 = 2 x 1000
#X4=np.matmul(EIGENVECT,X2)
else :
	U,S,V = np.linalg.svd(COV, full_matrices=False)
	X3=np.matmul(U.T[0:K],X1)


print(X3.shape) #ftanoume ston teliko pinaka, pou exei pali 2 diastaseeis


if K>=2:
	df = DataFrame(dict(x=X3[0,:], y=X3[1,:], label=y))
	colors = {0:'red', 1:'blue'}
	fig, ax = plt.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
		group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	plt.show()

if K==1:
	df = DataFrame(dict(x=X3[0,:], y=X3[0,:], label=y))
	colors = {0:'red', 1:'blue'}
	fig, ax = plt.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
		group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	plt.show()
# den vlepoume na kalutervei polu h eikona!


