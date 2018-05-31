import random
import numpy as np

file=open('motifs_in_sequence.fa')
sequences=[]
for f in file:
	sequences.append(list(f.strip())) #my sequences

def randomotive(seq,kappa):
	length=[x for x in range(0,len(seq)-kappa)] #subtracts a random kmer from a seq
	j=random.choice(length)
	kappamer=seq[j:j+kappa]
	return kappamer

def createPWM(PWMlist,kappa): #creates a PWM matrix out of kmer matrix
	PWM=[]
	for x in range(0,kappa):
		bases=[y[x] for y in PWMlist]
		PWM.append([['A','T','C','G'],[bases.count('A')/len(bases),bases.count('T')/len(bases),bases.count('C')/len(bases),bases.count('G')/len(bases)]])
	return PWM

	
def comparetoPWM(seq,PWM): #score of a kmer compared to a k-length PWM
	score=0
	for x in range(0,len(seq)):
		for y in range(0,4):
			if seq[x]==PWM[x][0][y]:
				score+=float(PWM[x][1][y])
	return score
def replaceseq(sequences,PWMlist,kappa,j,i): #replaces a kmer from the matrix with a certain kmer from the same seq 
	PWMlist[j]=sequences[j][i:i+kappa]
	

def replacerandomseq(sequences,PWMlist,kappa): #rrepalces a kmer with a random kmer form the sequence
	j=random.randint(0,len(sequences)-1)
	PWMlist[j]=randomotive(sequences[j],kappa)

def consensus(PWM): #creates the consensus seq from a PWM
	myseq=[]
	for j in PWM:
		index=(j[1].index(max(j[1])))
		myseq.append(j[0][index])
	return myseq
	
	
	
	
	
	
##########################################################################################################################################
	
	


mymotifs=[]
for w in range(2,18): #some kappas to try (we can extend that)
	kappa=int(w)
	PWMlist=[]
	for i in range(0,len(sequences)):
		PWMlist.append(randomotive(sequences[i],kappa)) #the PWMlist is the N x Kappa matrix of kmers
	TOTALscore=0
	counter=0
	while TOTALscore>=0:
		bestscore=0
		counter+=1
		j=random.randint(0,len(sequences)-1) #random seq
		PWM=createPWM([x for x in PWMlist if x!= PWMlist[j] ],kappa) #remove it from the list,create the PWM
		motifscore=[]
		motifpos=0
		for i in range(0,len(sequences[j])-kappa): #compare the removed seq with the PWM
			seqscore=comparetoPWM(sequences[j][i:i+kappa],PWM)
			if seqscore>=bestscore: #grab the best scoring kmer
				bestscore=seqscore
				motifpos=i
		replaceseq(sequences,PWMlist,kappa,j,motifpos) #replace it with the seq we removed
		if counter%1==0:
			replacerandomseq(sequences,PWMlist,kappa) #also replace a random kmer,to mix it up, avoid getting stuck
		if counter>=30000:#where to stop
			break
		print(kappa)
		print(consensus(PWM))
	for w in range(0,len(sequences)): #the score of the PWM for all of the seqs
		motifscore=[]
		for i in range(0,len(sequences[w])):
			seqscore=comparetoPWM(sequences[w][i:i+kappa],PWM)
			motifscore.append(seqscore/kappa)
		TOTALscore+=sum(motifscore)
	print(TOTALscore)
	mymotifs.append([TOTALscore,consensus(PWM),PWM])


#print(sorted(mymotifs))
infoz=[] 
counter=0
for x in range(0,len(mymotifs)): #to find the best kappa ??? (not sure)
	info=0
	for j in mymotifs[x][2]:
		for k in j[1]:
			info+=(np.log(k)*k)
	info=-info # information per residue
	counter+=1
	infoz.append([info/counter+2,counter]) # diairw me kappa gia na einai sugrishma
	
bestkappaindex=int(sorted(infoz)[-1][1]) # kalutero kappa?

print(mymotifs[bestkappaindex]) #exoume to kalutero kappa to score tou sthn allhlouxia , to consensus sequence kai to PWM tous(gia weblogo)

for j in range(0,len(mymotifs)):
	print(mymotifs[j][1]) #ola ta motiva pou vrika apo kappa=2:17 #paratiroume me to mati oti to GATA emfanizete suxna
