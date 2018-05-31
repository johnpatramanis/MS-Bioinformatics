#Copyright (C) 2017-2018 Alexandros Kanterakis, mail:kantale@ics.forth.gr
#url:https://gist.github.com/kantale/81d7d728c22fb35d77112c3633e17389



import argparse
import gzip
import re
import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
import concurrent.futures
from scipy.cluster import hierarchy

#python projectn1.py --vcf 1kgp_chr1.vcf.gz --sample_filename sample_information.csv --population GBR 200 --SNPs 100 --action SAMPLE_INFO SIMULATE VALIDATE_SAMPLE_INFO --output mysnps 
#python projectn1.py --input_filename mysnps --PCA_filename mypca --PCA_plot myplot --action PCA
#python projectn1.py --PCA_filename mypca --action CLUSTER
#python projectn1.py --vcf 1kgp_chr1.vcf.gz --sample_filename sample_information.csv --population GBR 100 --population FIN 100 --independent 100 --START 100 --END 50000000 --action FIND_RATIO
#python projectn1.py --vcf 1kgp_chr1.vcf.gz --sample_filename sample_information.csv --action DENDROGRAM


#########################Eisagwgh parametrwnnnn##########################################################################
parser = argparse.ArgumentParser()  
parser.add_argument('--vcf',nargs='+',type=str)
parser.add_argument('--sample_filename' ,nargs='+',type=str)
parser.add_argument('--pop', nargs='+')
parser.add_argument('--pca', nargs='+')
parser.add_argument('--action',nargs='+')
parser.add_argument('--population',action='append',nargs='+')
parser.add_argument('--SNPs',nargs='+')
parser.add_argument('--output',nargs='+')
parser.add_argument('--independent',nargs='+')
parser.add_argument('--input_filename',nargs='+')
parser.add_argument('--PCA_filename',nargs='+')
parser.add_argument('--PCA_plot',nargs='+')
parser.add_argument('--MINIMUM_AF',nargs='+')
parser.add_argument('--START',nargs='+',type=int)
parser.add_argument('--END',nargs='+',type=int)



args = parser.parse_args()
#########################################################################################################################

###########################dataset construction #########################################################################
def datamine(filename,sample_filename):
    """ By Lydia & Giannis 
	This functions requires  a filename corresponding to a .vcf filename as well as a  matching sample filename
	with the tags (Continent,Country,Population etc) that belong to each individual of the vcf file.
	Using these two it constructs a new .tsv file bearing the same name that contains the frequency of each population 
	based on the sample file tags.
	It requires to process line-by-line the entirety of the file so the process consumes a lot of time.
	The end product should have x+1 number of columns and n number of rows where x is the number of unique populations,
	n the number of SNPS in the file plus one more column with the possition of each SNP
	Args:VCF filename, its corresponding sample file name
    """
    samplefile = open(sample_filename) #opens the sample file
    sampledata=[]
    for line in samplefile:
        sampledata.append(line.strip().split('\t')) #reads each line and constructs the headers and the Areas list which contains all the different Populations
    sampleheaders=sampledata[0]
    sampledata=sampledata[1:] #2k population tags
    Areas=[]
    for x in sampledata:
        if x[1] not in Areas: #only the uniques
            Areas.append(x[1])

    data=[]
    i=0
    exported=open('{}.tsv'.format(filename),'w') #opens the vcf file
    for J in Areas:
        exported.write("{}\t".format(J)) #construct the headers of the new file (tsv)
    exported.write("POSITION")
    exported.write("\n")
    for line in file: #for each line ,if does not begin with ## and belongs to a SNP,
        if line[:2]!='##':
            data=line.strip().split('\t')
            if 'SNP' in data[7]:
                snpdata=data[9:]
                position=data[1]
                for y in Areas: #for each Population
                    j=[ snpdata[x] for x in range(0,len(snpdata)) if sampledata[x][1]==y] #checks if the tag of the sample belongs to the pop 
                    if i >=1: #because first line has headers
                        j="\t".join(j)
                        A=j.count("0|0")
                        B=j.count("1|0")+j.count("0|1")
                        C=j.count("1|1")
                        exported.write("{}\t".format((B+2*C)/len(j)*2)) #computes the frequency of the pop and adds it with a tab
                exported.write("{}".format(int(position))) #once all pop frequencies are done, adds the possition of the SNP and moves to the next one
                exported.write("\n")
                i+=1
                print(i) #to keep tabs of where we are
            if 'SNP' not in data[7]:
                print('non SNP detected and removed')
                i+=1

#########################################################################################################################
################################# VCF_INFO   ############################################################################
def vcf_info(file):
    """ By Lydia & Giannis 
	Counts the number of snps and samples that are found in the panda file
	Args: The sample file
    """
    for line in file: 
        if line[:2]!='##':
            data=line.strip().split('\t')
            numberofsamples=len(data[9:])
            break
    numberofSNPs=len(MYDATA.index)
    return numberofsamples,numberofSNPs
#########################################################################################################################
#########################################sample info#####################################################################
def sample_info(sample_file):
    """ By Lydia & Giannis 
	Creates a panda file from the sample file to answer the questions of Task 2)
	Prints all the information in a nice way: Each continent or super-pop and then the populations that belong to it
	Args: The sample file name
    """
    sample_file=open(sample_file,'r')
    samplenew=[x.split('\t') for x in sample_file]
    samplenew = [[y.strip() for y in x] for x in samplenew]
    samplenew[0]=samplenew[0][0:4]
    header=samplenew[0]
    sample_as_pd=pd.DataFrame.from_records(samplenew[1:],columns=header)

    areas=list(sample_as_pd["super_pop"].unique())
    print("File has",len(areas),"Areas.", )
    abc=list(sample_as_pd["super_pop"].unique()) #Contains all Supeprops - Continents
    for x in abc:
        print('Area {} is {} and contains {} samples, splitted in the following populations'.format(abc.index(x)+1,x,len(sample_as_pd[sample_as_pd["super_pop"]==x])))
        subpop=dict(sample_as_pd[sample_as_pd["super_pop"]==x]['pop'].value_counts())
        for y,z in subpop.items():  
            print(y,z,'samples') # All this to print it in the requested format!
    sample_file.close
    return sample_as_pd
#########################################################################################################################
#####################################VALIDATE SAMPLES####################################################################
def validate_samples(sample_file,filename):
    """ By Lydia & Giannis 
	Validates that all samples found in the vcf file are also found in the sample file
	Args: Sample filename , the vcf file name
    """
    samplefile=open(sample_file,'r')
    missingfromvcf=[]
    missingfromsamples=[]
    sampledata=[]
    sampleheaders=[]
    for line in samplefile:
        sampledata.append(line.strip().split('\t'))
        samples=[x[0] for x in sampledata[1:len(sampledata)]]
    for line in filename: 
        if line[:2]!='##':
            data=line.strip().split('\t')
            file=data[9:]
            break
    missingfromsamples=[x for x in file if x not in samples]
    missingfromvcf=[x for x in samples if x not in file]
    samplefile.close
    return missingfromsamples,missingfromvcf
#########################################################################################################################
#####################################Allele Creator######################################################################
def allele_creator(samples,frequency):
    """ By Lydia & Giannis 
	Validates that all samples found in the vcf file are also found in the sample file
	Args: an integer representing the number of individuals to be created, a float representing the frequency of the SNPS for the population 
    """
    samples=int(samples)
    majoralle='A'
    minoralle='B'
    majoralles=int(samples*2*frequency) #creates 2*population*frequency alleles
    minoralles=int(samples*2-majoralles) #the rest of the alleles
    alleles=majoralles*majoralle + minoralles*minoralle
    alleles=list(alleles)
    random.shuffle(alleles) #randomises the alleles for each individual
    myallels=[ alleles[x]+'/'+alleles[x+1] for x in np.arange(0,len(alleles)-1,2) ]
    return myallels #returns the population as list of individuals each repressented as a X/X

#########################################################################################################################
#################################GENOME to numbers Converter###############################################################################
   
def convert_genotype_data_to_numeral(genotypes):
    """ By Lydia & Giannis 
	Transforms a list of genotypes to a list of numericals representing them
	Args: A list of individuals each represented by a X/X
    """
    ret = []
    for snp in genotypes:
        if snp == 'A/A':
            ret.append(2)
        elif snp == 'A/B' or snp == 'B/A':
            ret.append(1)
        else:
            ret.append(0)

    return ret

#########################################################################################################################
#######################################PCA###############################################################################
def do_PCA(filename,outfilename,PCAname):
    """ By Lydia & Giannis 
	Using a file of Genotypes performs PCA and saves both the final matrix and its image.
	Args: File name containing the genotypes that we want to PCA, the outputfilename of the final matrix [Numberofindividuals x 2 ] and the output filename of the PCA image
    """
    inputfilename=open(filename,'r')
    outputfilename=open(outfilename,'w')
    populationinfo=open('{}.pops'.format(filename),'r') #assisting file created earlier(using the same file-name) that contains the name of each pop and the number of individuals in it
    pcapopulationinfo=open('{}.pops'.format(outfilename),'w') #assisting file created for later use, same purpose
    for n in populationinfo:
        pcapopulationinfo.write(n) #to keep our info for later use
        n=n.split('\t')
        popnames=[n[x] for x in np.arange(0,len(n)-1,2)] #population names
        popnumbers=[n[x] for x in np.arange(1,len(n),2)] #number of individuals in each pop
    genotypesnumberical=[] #will contain the numerical genotypes
    for l in inputfilename:
        L=convert_genotype_data_to_numeral(l.strip().split('\t'))
        genotypesnumberical.append(L)
    genotypes_array=np.array(genotypesnumberical)
    pca=PCA(n_components=2)
    pca.fit(genotypes_array.T)
    genotypes_PCA=pca.transform(genotypes_array.T) #PCA on the numerical genotypes
    for j in genotypes_PCA:
        outputfilename.write('{}'.format(j))
        outputfilename.write('\n')
    counter=0
    stoplist=[0]
    colorlist=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for j in popnumbers: #in order to asses where each pop begins and ends ,so we can paint them the right colour in the pca
        stoplist.append(int(j))
        stop=sum([x for x in stoplist]) #0->100->200->400 ... klp
        start=sum([int(x) for x in popnumbers])-sum([int(x) for x in popnumbers[counter:]]) #
        plt.plot(genotypes_PCA[start:stop,0],genotypes_PCA[start:stop,1],'.',color=colorlist[counter])
        counter+=1
    inputfilename.close
    outputfilename.close
    populationinfo.close
    pcapopulationinfo.close
    plt.savefig('{}.pdf'.format(PCAname))
#########################################################################################################################
##########################################CLUSTER########################################################################
def do_cluster(filename):
    """ By Lydia & Giannis 
	Checks how much each cluster matches the official clusters (labeled pops)
	Args: Filename of the pca
	"""
    inputfilename=open(filename,'r')
    populationinfo=open('{}.pops'.format(filename),'r') #assisting file with the names and number of individuals
    for n in populationinfo:
        n=n.split('\t')
        popnames=[n[x] for x in np.arange(0,len(n)-1,2)]
        popnumbers=[n[x] for x in np.arange(1,len(n),2)]
    genotypes_PCA=[]    
    for l in inputfilename: #a bit tricky to extract the numberical info of the PCA (matrix)
        h=re.search(r'(-*[0-9]+.[0-9]+)(-*[0-9]+.[0-9]+)',l)
        first=float(h.group(0))
        second=float(h.group(1))
        genotypes_PCA.append([first,second])
    genotypes_PCA=np.array(genotypes_PCA)
    kmeans = KMeans(n_clusters=len(popnumbers), random_state=0).fit(genotypes_PCA)
    stoplist=[0]
    counter=0
    errors=[0]
    for j in popnumbers: #in order to asses where each pop begins and ends ,so we can paint
        stoplist.append(int(j))
        stop=sum([x for x in stoplist]) #0->100->200->400 ... klp
        start=sum([int(x) for x in popnumbers])-sum([int(x) for x in popnumbers[counter:]])#start of each pop
        counter+=1
        popclust=[x for i,x in enumerate(kmeans.labels_) if start<int(i)<stop]#periexei ta labels gia ton sugekrimeno pop (px 1,2,1,1,0,1,0)
        clustcount=[]
        for f in np.unique(popclust):
            clustcount.append(popclust.count(f))#metrame poses fores vlepw to kathe label ston pop
        MYCLUST=np.unique(popclust)[clustcount.index(max(clustcount))]#to label pou vriskw pio suxna=> auto pou thewritika einai to label gia ton plhthusmo
        errors.append(sum([1 for i,x in enumerate(kmeans.labels_) if start<int(i)<stop and x!=MYCLUST]))#metraw posa dn exoun to swsto label
    errors= sum(errors)/sum([int(x) for x in popnumbers])#ola ta lathos labels/sunolo olwn twn pop
    return 'fail rate of {}%'.format(errors)
#########################################################################################################################   
def PCA_and_CLUSTER(genotypesnumberical,popnumbers):
    """ By Lydia & Giannis 
	Performs PCA and then checks to see how good the clustering is based on the pca
	Args: A list of numerical genotypes, the number of individuals in each pop (only takes two populations)
	"""
    genotypes_array=np.array(genotypesnumberical)
    pca=PCA(n_components=2)
    pca.fit(genotypes_array.T)
    genotypes_PCA=pca.transform(genotypes_array.T)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(genotypes_PCA)
    belong_to_1_clustered_1 = sum([1 for i,x in enumerate(kmeans.labels_) if i<int(popnumbers[0]) and x==1])
    belong_to_1_clustered_2 = sum([1 for i,x in enumerate(kmeans.labels_) if i<int(popnumbers[0]) and x==0])
    belong_to_2_clustered_1 = sum([1 for i,x in enumerate(kmeans.labels_) if i>=int(popnumbers[0]) and x==1])
    belong_to_2_clustered_2 = sum([1 for i,x in enumerate(kmeans.labels_) if i>=int(popnumbers[0]) and x==0])
    errors = belong_to_1_clustered_2 + belong_to_2_clustered_1
    errors= errors/sum([int(x) for x in popnumbers])
    return errors
############################################FIND RATIO#####################################################################
def find_ration(population,independent,flag):
    """ By Lydia & Giannis 
	The First time its called creates a selected number of Independent SNPs ,then each time add a certain number(10) of non-independent SNPS
	Args: List with both the names of the populations and the number of individuals in each,number of independent SNPs ,the flag signaling if its the first iteration or not
	"""
    if flag:
        for w in range(int(independent)):
            j=random.choice(myindexes)
            dataset.append(allele_creator(sum([int(population[y][1]) for y in range(0,len(population))]),np.mean([float(MYDATA[args.population[y][0]].iloc[j]) for y in range(0,len(args.population))])))
        flag=False
    else:
        h=0
        while h <=10:
            minimumfreq=0
            frequencies=[]
            j=random.choice(myindexes)
            frequencies.append(MYDATA[population[0][0]].iloc[j])
            frequencies.append(MYDATA[population[1][0]].iloc[j])
            pos=MYDATA['POSITION'].iloc[j]
            if args.MINIMUM_AF:
                minimumfreq=float(args.MINIMUM_AF[0])
            if max(frequencies)!=min(frequencies) and ((max(frequencies)+min(frequencies))/len(frequencies)>=minimumfreq) and (position[0]<=pos<=position[1]):
                #print(frequencies[0],frequencies[1])
                temp_list = []
                for y in population:
                    temp_list.extend(allele_creator(int(y[1]), float(MYDATA[y[0]][j])))
                temp_list=convert_genotype_data_to_numeral(temp_list)
                datasetnumerical.append(temp_list)
                h+=1
    #print(len(dataset))
    return flag
#########################################################################################################################
#########################################################################################################################
def find_ration_dendro(population, flag,increaser,increaser2):
    """ By Lydia & Giannis 
	The First time its called creates a certain number of Independent SNPs(1000) ,then each time adds a certain number of non-independent SNPS (10+ increaser 1,2 if they are active)
	Args: List with both the names of the populations and the number of individuals in each,the flag signaling if its the first iteration or not,the power of increaser 1,2
	"""
    if flag:
        for w in range(1000):
            j=random.choice(myindexes)
            dataset.append(allele_creator(200, np.mean([float(MYDATA[y].iloc[j]) for y in population])))
        flag=False
    else:
        h=0
        while h <= 10+increaser+increaser2: #how many to be created,increasers are used when popoulations are close to speed things up
            minimumfreq=0
            frequencies=[]
            j=random.choice(myindexes)
            frequencies.append(MYDATA[population[0]].iloc[j])
            frequencies.append(MYDATA[population[1]].iloc[j])
            if args.MINIMUM_AF:
                minimumfreq=float(args.MINIMUM_AF[0])
            if (max(frequencies)-min(frequencies))>=minimumfreq:
                #print(frequencies[0],frequencies[1])
                temp_list = []
                for y in population:
                    temp_list.extend(allele_creator(100, float(MYDATA[y][j])))
                temp_list=convert_genotype_data_to_numeral(temp_list)
                datasetnumerical.append(temp_list)
                h+=1
    return flag
#########################################################################################################################
#########################################################################################################################
if args.vcf:
    filename=args.vcf[0] 
    if re.search(r'gz',filename) != None: #anagnwrish arxeiou
        file = gzip.open(filename, 'rt')  #gia zip
    if re.search(r'gz',filename) == None: #gia oxi zip
        file=open(filename)

if args.sample_filename:
    sample_filename=str(args.sample_filename[0])
    try:
        MYDATA = pd.read_csv('{}.tsv'.format(filename), sep='\t')#koitaei ama uparxei to .tsv
    except FileNotFoundError:
        datamine(filename,sample_filename)
        MYDATA = pd.read_csv('{}.tsv'.format(filename), sep='\t') #an oxi to ftiaxnei kai to anoigei
    except NameError:
        print('please enter a valid vcf name and / or a valid sample name to begin')
    



if args.action:
    actions=list(args.action) #erwthma 2
    if 'VCF_INFO' in actions:
        print("The number of Samples is : {} and the number of SNPS is: {} ".format(vcf_info(file)[0],vcf_info(file)[1])) #vlepe funct vcf_info
    if 'SAMPLE_INFO' in actions: #vlepe funct sample_info
        sample_info(sample_filename)
    if 'VALIDATE_SAMPLE_INFO' in actions:
        missingall=validate_samples(sample_filename,file) #vlepe funct validate samples
        if len(missingall[0])==0 and len(missingall[1])==0: #kenes liste= kanena provlima
            print('Everything is ok!')
        if len(missingall[0])>0 and len(missingall[1])==0:
            print('sample/s {} were found in the sample file and missing from the vcf file'.format(missingall[0]))
        if len(missingall[0])==0 and len(missingall[1])>0:
            print('sample/s {} were found in the vcf file and missing from the sample file'.format(missingall[1]))
        if len(missingall[0])>0 and len(missingall[1])>0:
            print('\nThe sample/s {} were found in the sample file and missing from the vcf file\n'.format(missingall[0]))
            print('The sample/s {} were found in the vcf file and missing from the sample file'.format(missingall[1]))
    if 'SIMULATE' in actions and args.population and args.SNPs and args.output: 
        NUMBEROFSNPS=int(args.SNPs[0])
        outputfile=args.output[0]
        exported=open('{}'.format(outputfile),'w')#genotypes
        populationexport=open('{}.pops'.format(outputfile),'w')#additional info (which populations + how many from each)
        populationexport.write('\t'.join([(args.population[y][0]+'\t'+args.population[y][1]) for y in range(0,len(args.population))])) # info on what the file contains (population name followed by number of individuals in that population
        myindexes=list(MYDATA.index.values) #lista me ola ta pithana index
        for x in range(0,NUMBEROFSNPS): #poses fores tha trexei = me to posa snps theloume
            j=random.choice(myindexes) #random index = random SNP
            for y in range(0,len(args.population)): #gia kathepopulation
                exported.write('\t'.join(allele_creator(int(args.population[y][1]),float(MYDATA[args.population[y][0]][j])))) #opou args.poulation[1][0]=onoma pop,[1]=number of samples
                exported.write('\t')
            exported.write('\n')
        if args.independent: #to idio gia ta independent
            independent=args.independent[0]
            for w in range(0,int(independent)):
                j=random.choice(myindexes)
                exported.write('\t'.join(allele_creator(sum([int(args.population[y][1]) for y in range(0,len(args.population))]),np.mean([float(MYDATA[args.population[y][0]][j]) for y in range(0,len(args.population))]))))
                exported.write('\n')
                
    if 'PCA' in actions and args.input_filename and args.PCA_filename and args.PCA_plot: #ektelei thn do_PCA,vlepe funct
        do_PCA(args.input_filename[0],args.PCA_filename[0],args.PCA_plot[0])
        
        
        
        
    if 'CLUSTER' in actions and args.PCA_filename:
        print(do_cluster(args.PCA_filename[0]))#ektelei thn do_cluster, vlepe funct
        
        
    if 'FIND_RATIO' in actions:
        iterations=1#default option of iterations 
        if 'ITERATIONS' in actions: #an orisei o xrhsths
            finditerations=re.search(r'ITERATIONS([0-9]+)',''.join(actions))
            iterations=int(finditerations.group(1))
        for loops in range(0,iterations):
            flag=True #gia na ginoun prwta ta ind
            j=0 #to counter gia tis poses fores exei trexei to +non-independent
            k=0#counter gia ta errors pou upologizoume
            dataset=[]#to dataset mas sto opoio tha prosthetoume
            final=[]
            mean1000=[]#mazevoume ta errors gia na tous vgaloume to meso oro(arxika htan ana 1000, tlka ana 5)
            popnames=[args.population[x][0] for x in range(0,len(args.population))]#ta onomata twn plhthismwn
            popnumbers=[args.population[x][1] for x in range(0,len(args.population))]#posa atoma apo kathe plhthismo
            errors=float(0)
            totalerrors=[]
            myindexes=list(MYDATA.index.values)#ta indexes gia na ta epilegw(alliws an to kaname me randint(0,len(MYDATA)) petage error mia sto toso,mallon logo kapoiwn kenwn endiamesa)
            last=sorted([MYDATA['POSITION'].values])[-1][-1] #teleutaio position
            position=[0,last]#default positions edws twn opoiwn epilegoume SNPS
            if args.START: #An dwthei apo to xrhsth oria gia to position twn snps
                position[0]=args.START[0]
            if args.END:
                position[1]=args.END[0]
            print("Selecting SNPs between position {} and {} ".format(position[0],position[1]))
            totalerrors.append(0)#na exoume ena arxiko error gia na to sugrinoume
            datasetnumerical = []#to datasetmas numerical
            flag = find_ration(args.population,args.independent[0], flag)#1o run
            for x in dataset:#metatroph se numerical
                datasetnumerical.append(convert_genotype_data_to_numeral(x))
            while j>=0:#and off we go!
                j+=1
                flag = find_ration(args.population,args.independent[0], flag)#treximo tou function
                errors=PCA_and_CLUSTER(datasetnumerical,popnumbers)#upologismos error
                mean1000.append(errors)#ta mazevoume gia meso oro
                if j%5==0:
                    totalerrors.append(np.mean(mean1000))
                    mean1000=[]#upologismos mesou orou,adiasma
                    k+=1
                    print(totalerrors)#edw mazevoume ola ta errors gia na vlepoume thn poreia
                    if (totalerrors[k]-totalerrors[k-1]<=0.01 and totalerrors[k]<=0.03) or (len(totalerrors)>=100*int(args.independent[0]) and totalerrors[k]-totalerrors[k-1]<=0.05):#epilegoume na stamatisoume an exoune mikrh diafora metaxu tous ta errors K exoume ftasei se 97% success
                        percentage=(j*10)/((j*10)+int(args.independent[0])) #to success sunithos ftanei ekei koda (97%) kathos exoume ftiaxei me teteio tropo to find ratio ,alliws stamataei otan perasoume kabosa vimata kai den uparxei sovarh veltiwsh 
                        print('{} population specific SNPs were required to achieve population clarity'.format(j*10))
                        print('that\'s {}% of the total number of SNPs'.format(percentage))
                        final.append(int(j)*10)
                        break
                print('fail rate of {} '.format(errors))
        populationsmean=np.mean(final)
        print('The mean number of population specific SNPs for these 2 populations is {}'.format(populationsmean))



    if 'DENDROGRAM' in actions:
        sample_file=open(sample_filename,'r') # Pairnoume ola ta populations apo to VCF
        samplenew=[x.split('\t') for x in sample_file]#
        samplenew = [[y.strip() for y in x] for x in samplenew]#
        samplenew[0]=samplenew[0][0:4]#
        header=samplenew[0]#
        sample_as_pd=pd.DataFrame.from_records(samplenew[1:],columns=header)#
        areas=list(sample_as_pd["pop"].unique())#
        secure_random = random.SystemRandom()
        MYPOP =areas[0]# Dialegoume tuxaia enan gia na xekinisoume
        myindexes=list(MYDATA.index.values) #ola ta pithana snps gia na dialegoume tuxaia
        DENDRO=[] # H telikh lista me oles tis apostaseis, to distance matrix(trigwniko)
        DENDRO.append(MYPOP)
        PAIRS=[] #gia kathe plithismo mazevoume tis times (apostaseis) gia to kathe zeugos 
        USEDPOPS=[] #edw mazevoume osous plithismous exoume perasei hdh
        USEDPOPS.append(MYPOP)#vazoume tn prwto mas pop
        for dendro in range(0,len(areas)):#trexei 26 fores
            UNIQUES=[x for x in areas if x not in USEDPOPS]#edw einai osa menoun(kathe fora einai kata 1 mikrotero)
            for POP in UNIQUES:#gia enan enan pop kanoume thn sugrish me tn epilegmeno
                print(UNIQUES)#gia na vlepoume posoi menoun
                iterations=1#lets keep it simple :P
                for loops in range(0,iterations):
                    flag=True
                    j=0#####ta idia me thn apo panw
                    k=0
                    increaser=0#ama dn ftanoume grhgora se apotelesma autoi energopoioude
                    increaser2=0
                    extracounter=0#gia na metrame poses fores ta increasers exoun xrhsimopoihthei
                    extracounter2=0
                    mean1000=[]
                    dataset=[]
                    final=[]
                    popnames=[MYPOP,POP]#o dikos mas, me auton pou allazei kathe fora
                    popnumbers=[100,100]#simple and fair sugrish
                    errors=float(0)
                    totalerrors=[]
                    totalerrors.append(0)
                    flag=find_ration_dendro(popnames,flag,increaser,increaser2)
                    datasetnumerical = []
                    for x in dataset:
                        tmp=convert_genotype_data_to_numeral(x)
                        datasetnumerical.append(tmp) #mia apo ta idia me prin, prwta ta metatrepoume senumerical kai meta prosthetoume ta kainourgia SNPS se arithimitkh morfh gt einai pio grhgoro apo ta na ta metafrazoume ola mazi se kathe loopa
                    print(popnames)
                    while j>=0:
                        j+=1#mia apo ta idia me prin
                        flag=find_ration_dendro(popnames,flag,increaser,increaser2)#vlepe funct
                        errors=PCA_and_CLUSTER(datasetnumerical,popnumbers)
                        mean1000.append(errors)
                        if j%1==0:
                            totalerrors.append(np.mean(mean1000))
                            mean1000=[]
                            k+=1
                            print(totalerrors[k])
                            if len(totalerrors)>=200:#energopoihsh 1ou increaser
                                increaser=990
                                extracounter+=1
                                print('SNP increaser activated')
                            if len(totalerrors)>=250:#deuterou
                                increaser2=4000
                                extracounter2+=1
                                print('SNP super increaser activated')
                            if (totalerrors[k]-totalerrors[k-1]<=0.01 and totalerrors[k]<=0.05) or (len(totalerrors)>=300 and totalerrors[k]-totalerrors[k-1]<=0.05) or len(totalerrors)>=400:#pio apalo threshold apo prin
                                percentage=(j*10+extracounter*increaser+extracounter2*increaser2)/((j*10)+1000+extracounter*increaser+extracounter2*increaser2)
                                print('{} population specific SNPs were required to achieve population clarity'.format(j*25))
                                print('that\'s {}% of the total number of SNPs'.format(percentage))
                                final.append((1-percentage))
                                break
                        #print('fail rate of {} '.format(errors))
                populationsmean=np.mean(final)#to mean tou error (gia ama valoume polla iteratios)
                print('The mean number of population specific SNPs for these 2 populations ({}-{}) is {}'.format(MYPOP,POP,populationsmean))
                PAIRS.append(populationsmean)#to pairs exei tis times gia kathe zeugarh pop me ton pop mas
            if len(PAIRS)==0:
                break
            print(PAIRS,MYPOP)
            MYPOP=popnames[1]#next pop gia na ginei h sugrish
            USEDPOPS.append(popnames[1])
            DENDRO.append(PAIRS)#OLA ta zeugh
            PAIRS=[]       
        print(DENDRO)#behold
        all_populations = areas
        DENDRO.remove(DENDRO[0])# 1os pop me tn eauto tou ,dn exei timh,dn to theloume
        distancematrix=open('distancematrix.txt','w')
        for j in DENDRO:
            distancematrix.write(str(j))
            distancematrix.write('\n')			
        B= [DENDRO[x][y] for x in range(0,len(DENDRO)) for y in range(0,len(DENDRO[x]))]#pairnoume tous arithmous mas
        B= [float(x) for x in B]
        ytdist = np.array(B)
        Z = hierarchy.linkage(ytdist, 'single')#Let
        plt.figure(figsize=(15, 5))#the
        dn = hierarchy.dendrogram(Z, labels=all_populations)#magic
        plt.show()#commence