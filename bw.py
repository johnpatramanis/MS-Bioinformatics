print('welcome to the Burrows - Wheeler decoder! ')
question=str(input('would you like to insert you own sequnecess from a file?(Y/N) (else a test string from Maria Astrenaki will be used)  '))
if question=='Y':
    file=str(input('Please enter the name of the file of the string sequences (should be a plain text file with each a row a sequence ) '))
    seqs=[]
    f = open(file)
    for l in f:
    	seqs.append(str(l.strip()))

    
if question=='N':
    code='SI_$GNNAIoHall'
    
    
def decoder(code):
    L=list(code)
    F=sorted(L)
    truecode=[]
    for x in range(0,len(L)):
        if L[x]=='$':
            begin=x
    while len(truecode)<len(L):
        myF=F[begin]
        counter=0        
        for x in range(0,len(L)):
            if F[x]==myF:
                counter+=1
            if x==begin:
                break
        counter2=0
        for x in range(0,len(L)):
            if L[x]==myF:
                counter2+=1
            if L[x]==myF and counter==counter2:
                myL=L[x]
                begin=x
                break
        truecode.append(myL)
    return ''.join(truecode)
    
    
    
print('Here is/are your decoded string(/s)')    
if question=='N':
    print(decoder(code))
if question=='Y':
    for wow in seqs:
        print(decoder(wow))