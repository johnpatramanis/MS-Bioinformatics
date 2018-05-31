import numpy as np

print('welcome to the Needleman - Wunsch aligner! ')
seq1='ATTCGGCATT'
seq2='AAATTCGGCATTT'


gap=int(input('please insert your gap penalty e.g -2  '))
missm=int(input('please insert your missmatch penalty e.g -2  '))
match=int(input('please insert your match score e.g 2  '))


question=str(input('would you like to insert you own sequnecess?(Y/N) (else 2 test sequences will be used)  '))
if question=='Y':
	file=str(input('Please enter the name of the file of the sequences (should be a plain text file with each a row a sequence ) '))
	seqs=[]
	f = open(file)
	for l in f:
		seqs.append(str(l.strip()))
	seq1=seqs[0]
	seq2=seqs[1]

if question=='N':
	pass

if question!='Y' and question!='N':
	print('error - insert a valid answer (Y/N)')
seq1='-'+seq1 
seq2='-'+seq2




myarray=[ [0 for y in seq2] for x in seq1] # to matrix mas
increaserorig=gap
increaser=0

for x in range(0,len(seq1)): #gemizoume prwth gramh me penalties
	myarray[x][0]+=increaser
	increaser+=increaserorig
	
increaser=0
for y in range(0,len(seq2)): #gemizoume prwth colona me penalties
	myarray[0][y]+=increaser
	increaser+=increaserorig
	
for x in range(1,len(seq1)): #gemizoume ton pinaka me ta scores gia metakinisi apo kathe kouti
	for y in range(1,len(seq2)):
		previous=[ myarray[x-1][y],myarray[x][y-1],myarray[x-1][y-1]]
		myarray[x][y]+=max(previous)
		if x!=y:
			myarray[x][y]+=gap
		if seq1[x]!=seq2[y]:
			myarray[x][y]+=missm
		if seq1[x]==seq2[y]:
			myarray[x][y]+=match
			
print(np.asarray(myarray))

x=len(seq1)-1
y=len(seq2)-1
print(x,y)


align1=[]
align2=[]
align1.append(seq1[x])
align2.append(seq2[y])

while x!=0 and y!=0: #veltisto monopati apo telos
	bestmove=max([ myarray[x-1][y],myarray[x][y-1],myarray[x-1][y-1] ])
	if myarray[x-1][y]==bestmove:
		x-=1
		align1.append(seq1[x])
		align2.append('-')
	if myarray[x][y-1]==bestmove:
		y-=1
		align2.append(seq2[y])
		align1.append('-')
	if myarray[x-1][y-1]==bestmove:
		x-=1
		y-=1
		align1.append(seq1[x])
		align2.append(seq2[y])

print('Here are your sequences aligned')

		
print(align1[::-1])#ta tupwnoume anapoda (gt xekinisame apo to telos)
print(align2[::-1])

h=open('aligned.txt','w') #ta kanoume kai ena save se ena arxeio
for r in align1[::-1]:
	h.write(str(r))
h.write('\n')
for r in align2[::-1]:
	h.write(str(r))
