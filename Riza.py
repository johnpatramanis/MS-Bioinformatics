import numpy as np
import random
mynumb=random.uniform(0,1000)
myrange=[-1*mynumb,mynumb]
print('Our randomly selected space is {}'.format(myrange))
def equation(x): #h exiswsh mou
	return 2*(x**2)-(5*x)-12
def equationparagog(x):
	return 4*x-5
################# BINARY SEARCH ##############	

ans=(max(myrange)+min(myrange))/2 #to endiameso shmeio , to 0
spacer=0.001
final=[]
counter=0
rangeA=[myrange[0],ans]
rangeB=[ans,myrange[1]]
high = max(rangeB)
low = min(rangeB)
ans=(high+low)/2

ans+=spacer
while abs(equation(ans))-spacer>0.1:
	ans+=spacer
	if equation(ans)>0 and equation(high)>0:
		high=ans
		ans=(high+low)/2
		counter+=1
	if equation(ans)<0 and equation(high)>0:
		low=ans
		ans=(high+low)/2
		counter+=1
final.append(float(ans))

high = min(rangeA)
low = max(rangeA)
ans=(high+low)/2
ans+=spacer
while abs(equation(ans))-spacer>0.1:
	ans+=spacer
	if equation(ans)>0 and equation(low)>0:
		low=ans
		ans=(high+low)/2
		counter+=1
	if equation(ans)<0 and equation(high)<0:
		high=ans
		ans=(high+low)/2
		counter+=1
final.append(float(ans))
print('The answers to our equation are {} \n and it took algorythm {} ticks to solve it'.format(final,counter))

#############################################NEWTON RAPHSON METHOD ####################################
counter2=0
final2=[]
ans=mynumb
while abs(equation(ans))-spacer>0.1:
	ans+=spacer
	if equation(ans)>0:
		counter2+=1
		ans=ans-(equation(ans)/equationparagog(ans))
	if equation(ans)<0:
		counter2+=1
		ans=ans+(equation(ans)/equationparagog(ans))
final2.append(ans)
ans=-1*mynumb
while abs(equation(ans))-spacer>0.1:
	ans+=spacer
	if equation(ans)>0:
		counter2+=1
		ans=ans-(equation(ans)/equationparagog(ans))
	if equation(ans)<0:
		counter2+=1
		ans=ans+(equation(ans)/equationparagog(ans))
final2.append(ans)
print('The answers to our equation are {} \n and it took algorythm {} ticks to solve it'.format(final2,counter2))
# vlepoume pws h deuterh methodos einai pio apaithitkh se xrono apo thn 1h 