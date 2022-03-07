from itertools import count
from sre_constants import SUCCESS
import time
import numpy as np
import random as rand
import os
import math
import matplotlib.pyplot as plt
import pylab

NUMBERS_TO_DETECT = 10
SEUIL_ACTIVATION = 0.5
TAUX_APPRENTISSAGE = 0.1
NUMBER_OF_DATA_LEARN_FILE = 10
EPSILON_ERROR = 0.1


class dataLearn:
    def __init__(self):
        self.input = []
        self.officialResult = -1
        self.lignes = []
        self.nbLines =0
        self.arrayOfficialResult = [0]*NUMBERS_TO_DETECT

class neuron:
    def __init__(self) :
        self.weightLayer = [] #la liste des poids qui y sont associés
        self.value = 0 # le résultat présent pour ce neuron        
        self.errorDetect = 0
        self.potentiel = 0
        self.potentielSigmoid = 0    


def init_neurons(nombrePoids,nombreNeuron):
    neurons = []
    for i in range(nombreNeuron):
        newNeuron = neuron()
        initWeight(newNeuron,nombrePoids)
        neurons.append(newNeuron)
    return neurons

def initWeight(neuron,nombrePoids) :
    neuron.weightLayer = []
    for i in range(nombrePoids):
        neuron.weightLayer.append(rand.random()/nombrePoids)
        # neuron.weightLayer.append(1)




def init_data(dataFileIndex,learn=True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFile = open(dir_path+"/data_learn_10/data"+str(dataFileIndex)+".txt", "r")
    lines = dataFile.readlines()
    dataFile.close() 
    j=0
    i=0
    data = dataLearn()
    data.lignes = lines
    numberLines = len(lines)
    data.nbLines = numberLines
    # print("number line : "+str(numberLines))
    for x in range(0,numberLines-1):
        line = lines[x]
        lenOfLine = len(line)-1
        for j in range(0,lenOfLine):
            if(line[j] =='.'):
                charToNumber = 0
            else :
                charToNumber = 1 
            data.input.append(charToNumber)
            i+=1
    data.officialResult = int(lines[numberLines-1])
    data.arrayOfficialResult[data.officialResult] =1 #On met un 1 pour la case d'arrivée
    return data




#Must return the index of the best neuron
def calculateOfficialResult(lastLayer):
    bestNeuron = 0
    bestResult = -100
    i= 0
    for neuron in lastLayer:
        if(neuron.potentiel>bestResult):
            bestNeuron = i
            bestResult = neuron.potentiel
        i+=1
    return bestNeuron    







def calculatePropagation_1(input,neurons):
    length =len(input)

    for neuron in neurons:
        neuron.potentiel = 0
        for i in range(0,length):
            neuron.potentiel += input[i]*neuron.weightLayer[i]
        neuron.value = sigmoid(neuron.potentiel)
    return neurons

def calculatePropagation_2(neuronsInput,neurons):
    length =len(neuronsInput)
    for neuron in neurons:
        neuron.potentiel = 0
        for i in range(0,length):
            neuron.potentiel += neuron.weightLayer[i]*neuronsInput[i].value
        neuron.value = sigmoid(neuron.potentiel)
    return neurons

#Effectue une propagation, en calculant avec les datas/poids actuels
#Retourne le resultat de la propagation
def calculatePropagation(data,firstLayer,secondLayer):
    firstLayer = calculatePropagation_1(data.input,firstLayer)
    secondLayer = calculatePropagation_2(firstLayer,secondLayer)
    return firstLayer,secondLayer


# Fonction d'activation
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

# Dérivée de la fonction d'activation
def sigmoidPrime(value):
    return value * (1 - value)



#Applique la correction des poids
def ApplicateCorrectionToWeight(data,firstLayer,secondLayer):
    indexOfNeuron=0

    for neuron in secondLayer:
        # neuron.errorDetect = sigmoidPrime(neuron.value)*data.arrayOfficialResult[indexOfNeuron]-neuron.potentiel
        neuron.errorDetect = sigmoidPrime(neuron.value)*(data.arrayOfficialResult[indexOfNeuron]-neuron.value)
        neuron.potentielSigmoid = sigmoidPrime(neuron.value)
        indexOfNeuron+=1


    indexOfNeuronInFirstLayer = 0
    for neuron in firstLayer:
        sommeOfErrorLastLayer = 0
        for neuronLastLayer in secondLayer:       
            sommeOfErrorLastLayer += neuronLastLayer.errorDetect*neuronLastLayer.weightLayer[indexOfNeuronInFirstLayer]
        neuron.errorDetect = sigmoidPrime(neuron.value)*sommeOfErrorLastLayer
        indexOfNeuronInFirstLayer+=1

    
    for neuron in secondLayer:
        for i in range(len(neuron.weightLayer)):
            neuron.weightLayer[i] += TAUX_APPRENTISSAGE*neuron.errorDetect*firstLayer[i].potentiel
        

    for neuron in firstLayer:
        for i in range(len(neuron.weightLayer)):
            neuron.weightLayer[i] += TAUX_APPRENTISSAGE*neuron.errorDetect*data.input[i]


    return firstLayer,secondLayer


#Return true if all result of data set are good
# if one or more fail, return False
def AreAllTestSuccess(list):
    for i in range(0,NUMBER_OF_DATA_LEARN_FILE):
        if(list[i]==False):
            return False
    return True

#Permet de calculer l'erreur totale 
def CalculateErrorTotal(data,lastLayer):
    indexNeuron = 0
    erreurCalcule = 0
    for neuron in lastLayer: 
        erreurCalcule += abs(data.arrayOfficialResult[indexNeuron]-neuron.value)
        indexNeuron+=1
    return erreurCalcule


def runDataLearnMethod2(firstLayer,lastLayer):
    listError = [False]*NUMBER_OF_DATA_LEARN_FILE 
    counter = 1
    evolutionOfFail = []
    countFail = printNumberSuccessAndFail(listError)
    evolutionOfFail.append(countFail)
    
    error = NUMBER_OF_DATA_LEARN_FILE
    evolutionOfError = []
    while  error>EPSILON_ERROR and counter<10000: #on fixe une limite en cas de boucle trop longue
        for i in range(0,NUMBER_OF_DATA_LEARN_FILE): #Apprend sur tout le set
            data = init_data(i)
            firstLayer,lastLayer = calculatePropagation(data,firstLayer,lastLayer)
            firstLayer,lastLayer = ApplicateCorrectionToWeight(data,firstLayer,lastLayer) # On applique la correction des poids 
            
        error= 0
        for i in range(0,NUMBER_OF_DATA_LEARN_FILE): #Calcul les erreurs restantes sans modifier les poids
            data = init_data(i)
            firstLayer,lastLayer = calculatePropagation(data,firstLayer,lastLayer)
            listError[i]= ( calculateOfficialResult(lastLayer)==data.officialResult)
            error += CalculateErrorTotal(data,lastLayer)
            # print("Résultat trouvé : ",calculateOfficialResult(lastLayer)," Resultat officiel : ",data.officialResult)
            
            countFail = printNumberSuccessAndFail(listError)
            evolutionOfFail.append(countFail)

        print("Counter :",counter," error : ",error)
        evolutionOfError.append(error)
        counter +=1
    print("nombre iteration : ",counter)
    return firstLayer,lastLayer,evolutionOfError


# print the number of success and fail
#return the number of fail
def printNumberSuccessAndFail(listError):
    countSuccess =0
    countFail =0
    for i in listError:
        if(i):
            countSuccess +=1
        else:
            countFail +=1
    
    # print("Nb Success : ",countSuccess," Nb Fail : ",countFail)
    
    return countFail

#Correspond à la partie généralisation
#Effectue 50 itérations sur un jeu de motifs bruités
#retourne le nombre d'erreur
def run50timesOnDataModified(data,tauxBruit,neurons):
    
    nbIteration = 50
    listError = [False]*nbIteration 
    counter = 1
    for i in range(0,nbIteration):
        dataModified = bruitage(data,tauxBruit)
        neurons = calculatePropagation(dataModified,neurons)                
        listError[i]= ( calculateOfficialResult(neurons)==data.officialResult)
        counter +=1
    counterOfError = printNumberSuccessAndFail(listError)
    
    return counterOfError

def courbeGeneralisationMotif(integerOfData,neurons):
    evolutionOfFail = []
    data = init_data(integerOfData)
    for i in range(0,100):
        counterOfFail = run50timesOnDataModified(data,i,neurons)
        evolutionOfFail.append(counterOfFail*2)
        

    return evolutionOfFail


def courbeGeneralisation(neurons):
    
    for i in range(NUMBERS_TO_DETECT):
        evolutionOfFail =courbeGeneralisationMotif(i,neurons)            
        plt.plot(evolutionOfFail,label=("Courbe de generalisation Motif ",i))
    plt.xlabel("Taux de bruitage")
    plt.ylabel("Pourcentage erreur")
    plt.legend()

def plotCout(listeCout):
    plt.plot(listeCout)   


#Permet de bruiter une image avec un certain pourcentage
def bruitage(data,pourcentageDeBruit):
    dataModified = dataLearn()
    dataModified.lignes = data.lignes.copy()
    dataModified.nbLines = data.nbLines
    dataModified.officialResult = data.officialResult
    dataModified.input = data.input.copy()
    counterOfPixelModified = 0
    i = 0
    nbPixelToModify = len(dataModified.input)*pourcentageDeBruit/100
    
    while counterOfPixelModified < nbPixelToModify :
        i = rand.randrange(len(data.input))
        counterOfPixelModified +=1
        if dataModified.input[i] == 0:
            dataModified.input[i]=1
        else :
            dataModified.input[i]=0
        
        
    return dataModified







    
        
#permet d'initialiser en chargeant un premier fichier et d'initialiser un certain nombre de neurones
def init(nbPoids):
    # data = init_data(0)
    # nbPoids = len(data.input)
    return init_neurons(nbPoids)


def init_neuron_test(nombrePoids):
    neurons = []
    NombreNeuronSortie = NUMBERS_TO_DETECT
    for i in range(NombreNeuronSortie):
        newNeuron = neuron()
        initWeight_test(newNeuron,nombrePoids)
        neurons.append(newNeuron)
    return neurons

def initWeight_test(neuron,nombrePoids) :    
    neuron.weightLayer = []
    for i in range(nombrePoids):
        neuron.weightLayer.append(0.125)
def initTest():
    data = init_data(0)
    nbPoids = len(data.input)
    return init_neuron_test(nbPoids)


def runTest():
    neurons = initTest()
    print("test init finish")
    runDataLearnMethod2(neurons)

#Permet de sauvergarder l'ensemble des poids dans un fichier pour une lecture future
def writeNeuronWeightOnFile(neurons,index):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFile = open(dir_path+"/save_neurons/save"+str(index)+".txt", "w")

    for neuron in neurons:
        dataFile.write(str(neuron.weightLayer))    
        dataFile.write("\n")
    dataFile.close()
    

def runAllFileToTest(neurons):
    listError = [False]*NUMBER_OF_DATA_LEARN_FILE
    evolutionOfFail = []
    countFail = printNumberSuccessAndFail(listError)
    evolutionOfFail.append(countFail)
    error = NUMBER_OF_DATA_LEARN_FILE
    evolutionOfError = []
    evolutionOfError.append(error)
    for i in range(0,NUMBER_OF_DATA_LEARN_FILE): #Apprend sur tous le set
        data = init_data(i)
        neurons = calculatePropagation(data,neurons)
        print("Résultat trouvé : ",calculateOfficialResult(neurons)," Resultat officiel : ",data.officialResult)
        listError[i]= ( calculateOfficialResult(neurons)==data.officialResult)
    return neurons


def PrintManagerEvolutionOfError(evolutionOfError,title):
    plt.plot(evolutionOfError,label=title)
    plt.xlabel("Nombre itérations")
    plt.ylabel("Erreur")    
    plt.legend()
    pylab.show()





data = init_data(0)

nombreNeuronCouche1 = 100
nombrePoidsParNeuronCouche1 = len(data.input)
neuronsFirstLayer = init_neurons(nombrePoidsParNeuronCouche1,nombreNeuronCouche1)

nombreNeuronCouche2 = NUMBERS_TO_DETECT
nombrePoidsParCouche2 = nombreNeuronCouche1
neuronsSecondLayer = init_neurons(nombrePoidsParCouche2,nombreNeuronCouche2)

# data = init_data(10)
# firstLayer,lastLayer = calculatePropagation(data,neuronsFirstLayer,neuronsSecondLayer)

neuronsFirstLayer,neuronsSecondLayer,evolutionOfError = runDataLearnMethod2(neuronsFirstLayer,neuronsSecondLayer)
# # writeNeuronWeightOnFile(neurons,0)
#runAllFileToTest(neurons)
# # runTest()

title = "Evolution de l'erreur avec epsilon="+str(EPSILON_ERROR)
PrintManagerEvolutionOfError(evolutionOfError,title)
# courbeGeneralisation(neurons)

print("FIN")
# init()
# runDataLearn()

# courbeGeneralisation()

pylab.show()

