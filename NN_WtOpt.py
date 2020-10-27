from sklearn.datasets import load_iris
from sklearn import preprocessing as pp
from sklearn import utils
from sklearn import metrics
from statistics import mean
import numpy as np
import scipy
import matplotlib.pyplot as plt

def initial_pop(input_neurons, hidden_neurons, output_neurons, pop_size,low,high):

    params = [dict() for x in range(pop_size)]

    for x in range(pop_size):
        wH = np.random.uniform(low, high, size=(
            hidden_neurons, input_neurons))  # weights of hidden 
        wO = np.random.uniform(low, high, size=(
            output_neurons, hidden_neurons))  # weights of output
        bH = np.random.uniform(low, high, size=(
            hidden_neurons, 1))  # bias of hidden   
        bO = np.random.uniform(low, high, size=(
            output_neurons, 1))  # bias of output

        params[x] = {"wH": wH, "wO": wO, "bH": bH, "bO": bO}

    return params

# Function to compute the losses for the population on the NN
def get_loss_accuracy(curGen,pop_size,X,Y):

    losses = [float() for x in range(pop_size)]
    accuracy = [float() for x in range(pop_size)]

    for x in range(pop_size):
        wH = curGen[x]["wH"]
        bH = curGen[x]["bH"]
        zH = np.dot(wH, X.T) + bH

        aH = np.maximum(zH, 0)  # output of hidden layer

        # output layer
        wO = curGen[x]["wO"]
        bO = curGen[x]["bO"]
        zO = np.dot(wO, aH) + bO

        zO = zO.T  # (m, output_neurons)

        # Apply softmax to it
        aO = scipy.special.softmax(zO)
        
        # Stores the index of the maximum prob out of the output
        y_pred = np.argmax(aO, axis=1)

        # binarize the labels with fixed classes
        y_pred = pp.label_binarize(y_pred, classes=[0, 1, 2])

        # calculate the accuracy and loss
        accuracy[x] = metrics.accuracy_score(Y, y_pred)
        losses[x] = metrics.log_loss(Y, aO)

    return (losses, accuracy)

# Function to calculate the fitness of the population
def get_fitness(losses, pop_size):

    fitness = [float() for x in range(pop_size)]

    invertedLosses = [1/x for x in losses]
    sum = np.sum(invertedLosses)

    for x in range(pop_size):
        fitness[x] = (invertedLosses[x]/sum)*100

    return fitness

# Function to generate two parents for Roulette Wheel selection
def roulette_wheel(fitness, pop_size):

    # get 2 parents without replacement with cumulative frequency
    count = 0
    p1 = -1
    p2 = -1
    while count != 2:
        val = np.random.random_sample()*100 

        cf = 0
        i = 0 
        while cf < val and i < pop_size:
            cf = cf + fitness[i]
            i = i + 1

        if p1 == -1:
            p1 = i 
            count = count + 1
        elif p2 == -1 and p1 != i:
            p2 = i  # to store the second parent and must not be same as first
            count = count + 1

    return (p1-1, p2-1)

def flattenChromo(chromoMat):    
    chromoFlat = []
    for key in chromoMat:
        chromoFlat=np.concatenate((chromoFlat,chromoMat[key].flatten()))
    return chromoFlat
        
def unFlattenChromo(chromoFlat,iCtr,hCtr,oCtr):

    temp=np.split(chromoFlat,[hCtr*iCtr])
    wHflat=temp[0]
    chromoFlat=temp[1]
    
    temp=np.split(chromoFlat,[oCtr*hCtr])
    wOflat=temp[0]
    chromoFlat=temp[1]

    temp=np.split(chromoFlat,[hCtr])
    bHflat=temp[0]
    chromoFlat=temp[1]
    
    bOflat=chromoFlat
    
    wH=np.array(wHflat).reshape(hCtr,iCtr)
    wO=np.array(wOflat).reshape(oCtr,hCtr)
    bH=np.array(bHflat).reshape(hCtr,1)
    bO=np.array(bOflat).reshape(oCtr,1)

    chromoMat = {"wH": wH, "wO": wO, "bH": bH, "bO": bO}

    return chromoMat

def flattenGen(curGen):

    curGenFlat = []

    for chromoMat in curGen:
        chromoFlat = flattenChromo(chromoMat)
        curGenFlat.append(chromoFlat)

    return curGenFlat

def unFlattenGen(curGenFlat,iCtr,hCtr,oCtr):

    curGen = []    

    for chromoFlat in curGenFlat:
        chromoMat = unFlattenChromo(chromoFlat,iCtr,hCtr,oCtr)
        curGen.append(chromoMat)

    return curGen


def onePointCrossover(p1,p2,tot_genes):
    
    point=np.random.randint(1,tot_genes)
    
    off1=np.concatenate((p2[0:point],p1[point:]))
    off2=np.concatenate((p1[0:point],p2[point:]))
    
    return off1,off2

def onePointMutation(chromo,tot_genes,low,high,ProbOfMut):
    
    for i in range(tot_genes):
        randNum=np.random.rand()
        if randNum < ProbOfMut:
            chromo[i] = float(np.random.uniform(-1.0,1.0,(1,1))) 
    
    return chromo

def createNextGen(curGenFlat,fitness,popLoss,pop_size,tot_genes,ProbOfCross,ProbOfMut,low,high):

    elitePop = np.argsort(popLoss)[0:4]
    
    #print(elitePop)
    #print(curGenFlat)
    
    nextGenFlat = curGenFlat    
    i=0
    while i<4:
        nextGenFlat[i]=curGenFlat[elitePop[i]]  
        #print(round(popLoss[elitePop[i]],3)," ",end="")
        #print(nextGenFlat[i])
        i=i+1

    while i<pop_size:
        p1,p2 = roulette_wheel(fitness, pop_size)
        
        nextGenFlat[i]=curGenFlat[p1]
        nextGenFlat[i+1]=curGenFlat[p2]

        randNum=np.random.rand()
        if randNum<ProbOfCross:
            off1, off2 = onePointCrossover(curGenFlat[p1],curGenFlat[p2],tot_genes)
            nextGenFlat[i]=off1
            nextGenFlat[i+1]=off2
        
        nextGenFlat[i]=onePointMutation(nextGenFlat[i],tot_genes,low,high,ProbOfMut)
        nextGenFlat[i+1]=onePointMutation(nextGenFlat[i+1],tot_genes,low,high,ProbOfMut)    
        
        i=i+2

    #print(nextGenFlat)
    return nextGenFlat


def plot_Metrics(i,metricDict,metricName,maxGen,xStep,yStep):
    
    plt.figure(i)

    xMaxVal=maxGen+1
    yMaxVal=max(metricDict['max'])+1
    
    plt.xlim(0,xMaxVal)
    plt.ylim(0,yMaxVal)
    
    plt.xticks(np.arange(0,xMaxVal,xStep))
    plt.yticks(np.arange(0,yMaxVal,yStep))
    
    plt.xlabel("Generation No.")
    plt.ylabel(metricName)
    plt.title("NN %s vs Generation"%(metricName))
    
    maxColor="red"
    minColor="green"
    meanColor="blue"
    """ 
    m1,=plt.plot(metricDict['max'],marker="x",color=maxColor)
    m2,=plt.plot(metricDict['min'],marker="+",color=minColor)
    m3,=plt.plot(metricDict['mean'],marker="^",color=meanColor)
     """

    m1,=plt.plot(metricDict['max'],color=maxColor)
    m2,=plt.plot(metricDict['min'],color=minColor)
    m3,=plt.plot(metricDict['mean'],color=meanColor)
    
    plt.legend([m1,m2,m3],["Max "+metricName,"Min "+metricName,"Average "+metricName],loc="upper right")
    
    return



#>>>>>>>>>>>>>>>>>>>>>>>>>>MAIN>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

'''
ARCHITECTURE OF NN ---
Number of input neurons = 4
Number of hidden layers = 1
Number of hidden neurons = 4
Number of output neurons = 3

Total parameters = 4*4 + 4*3 + 4 + 3 = 16 + 12 + 6 = 35
'''

################# Dataset Preprocessing ################

dataset = load_iris()
X = dataset["data"] 
Y = dataset["target"]

input_neurons = X.shape[1]
hidden_neurons = 4
output_neurons = np.unique(Y).shape[0]

encoder = pp.LabelBinarizer()
Y = encoder.fit_transform(Y)
X_shuffle, Y_shuffle = utils.shuffle(X, Y)

################# Defining Hyperparameters ################

tot_genes= input_neurons*hidden_neurons+hidden_neurons*output_neurons+hidden_neurons+output_neurons
pop_size = 30
low, high = -2.0, 2.0
ProbOfCross=0.8
ProbOfMut=0.01
maxGen=30

# Dictionary of Lists to store population metrics for plotting graphs afterwards
LossLines = {'max':[],'min':[],'mean':[]}
FitnessLines = {'max':[],'min':[],'mean':[]}
AccuracyLines = {'max':[],'min':[],'mean':[]}

################# Creating the Gen 0 Population ################

print("\n  Generating initial population...")
initPop = initial_pop(input_neurons, hidden_neurons, output_neurons, pop_size,low,high)
curGen=initPop

################# Creating further generations using GA ################

print("\n  Creating next generations...")
for i in range(0,maxGen):
    
    ProbOfMut = 1.0/(i+10)
    losses, accuracy = get_loss_accuracy(curGen,pop_size,X_shuffle,Y_shuffle)
    fitness = get_fitness(losses,pop_size)

    #print([round(e,3) for e in fitness])
    #print("",[round(e,3) for e in np.sort(losses)])
    #print("",[round(e,3) for e in losses])

    curGenFlat = flattenGen(curGen)
    
    nextGenFlat=createNextGen(curGenFlat,fitness,losses,pop_size,tot_genes,ProbOfCross,ProbOfMut,low,high)
    
    nextGen = unFlattenGen(nextGenFlat,input_neurons, hidden_neurons, output_neurons)
    
    print("\n\tGen %d Done!"%(i+1),end="")
    curGen = nextGen

    LossLines['max'].append(max(losses))
    LossLines['min'].append(min(losses))
    LossLines['mean'].append(mean(losses))

    FitnessLines['max'].append(max(fitness))
    FitnessLines['min'].append(min(fitness))
    FitnessLines['mean'].append(mean(fitness))
    
    AccuracyLines['max'].append(max(accuracy)*100)
    AccuracyLines['min'].append(min(accuracy)*100)
    AccuracyLines['mean'].append(mean(accuracy)*100)

print("\n\tDone!!!")

xStep=100
plot_Metrics(0,LossLines,"Loss",maxGen,xStep,1)
plot_Metrics(1,FitnessLines,"Fitness",maxGen,xStep,1)
plot_Metrics(2,AccuracyLines,"Accuracy",maxGen,xStep,5)

plt.show()
#plt.pause(2)




'''

def plot_Population(figNum,title,params,pop_size):

    plt.figure(figNum)
    plt.xlim(0,31)
    plt.xticks(np.arange(1,31,1))
    plt.xlabel("Population No.")
    plt.ylabel("Weight Value")
    plt.title(title)
    l1color="#b19cd9"
    l2color="orange"
    m1,=plt.plot([],[],marker="x",color=l1color)
    m2,=plt.plot([],[],marker="+",color=l2color)
    plt.legend([m1,m2],["First Layer","Second Layer"],loc="upper right")
    plt.ion()
    plt.show()

    l1color="#b19cd9"
    l2color="orange"    
    markerDict={"wH": "x", "wO": "+", "bH": "x", "bO": "+"}
    colorDict={"wH": l1color, "wO": l2color, "bH": l1color, "bO": l2color}
    labelDict={"wH": "First Layer", "wO": "Second Layer", "bH": "First Layer", "bO": "Second Layer"}
    
    for i in range(0,pop_size):        
        for key in params[i]:
            weightList=params[i][key].flatten()
            x=[i+1]*len(weightList)
            plt.scatter(x,weightList,marker=markerDict[key],color=colorDict[key],label=labelDict[key])

    plt.ioff()
    return

'''