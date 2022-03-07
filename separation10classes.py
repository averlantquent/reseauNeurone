from itertools import count
from sre_constants import SUCCESS
import time
import numpy as np
import random as rand
import os
import math
import matplotlib.pyplot as plt
import pylab



class dataLearn:
    def __init__(self):
        self.input = []
        self.officialResult = -1
        self.lignes = []
        self.nbLines =0
        self.arrayOfficialResult = [0]*NUMBERS_TO_DETECT

class neuron_layer:
    def __init__(self) :
        self.weightLayer = [] #la liste des poids qui y sont associés
        self.value = 0 # le résultat présent pour ce neuron
        

NUMBERS_TO_DETECT = 10
SEUIL_ACTIVATION = 0.5
TAUX_APPRENTISSAGE = 0.1
NUMBER_OF_DATA_LEARN_FILE = 10
EPSILON_ERROR = 0.01

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
def calculateOfficialResult(neurons):
    bestNeuron = 0
    bestResult = -100
    i= 0
    for neuron in neurons:
        if(neuron.value>bestResult):
            bestNeuron = i
            bestResult = neuron.value
        i+=1
    return bestNeuron    

#Effectue une propagation, en calculant avec les datas/poids actuels
#Retourne le resultat de la propagation
def calculatePropagation(data,neurons):
    
    length =len(data.input)
        
    for neuron in neurons:
        neuron.value = 0
        for i in range(0,length):
            neuron.value += data.input[i]*neuron.weightLayer[i]
    return neurons

#retourne le résultat de la sigmoid sur l'argument d'entré
def sigmoid(result ):
    result = 1 / (1 + math.exp(-result))
    return result

 

#Applique la correction des poids
def ApplicateCorrectionToWeight(data,neurons):   
    indexOfNeuron=0
    for neuron in neurons:

        
        erreurCalcule =  data.arrayOfficialResult[indexOfNeuron]-neuron.value
        i= 0
        for i in range(len(neuron.weightLayer)):
            neuron.weightLayer[i] +=TAUX_APPRENTISSAGE*erreurCalcule*data.input[i]
        indexOfNeuron+=1
    return neurons



#Return true if all result of data set are good
# if one or more fail, return False
def AreAllTestSuccess(list):
    for i in range(0,NUMBER_OF_DATA_LEARN_FILE):
        if(list[i]==False):
            return False
    return True

#Permet de calculer l'erreur totale 
def CalculateErrorTotal(data,neurons):
    indexNeuron = 0
    erreurCalcule = 0
    for neuron in neurons: 
        erreurCalcule += abs(data.arrayOfficialResult[indexNeuron]-neuron.value)
        indexNeuron+=1
    return erreurCalcule


def runDataLearnMethod2(neurons):
    listError = [False]*NUMBER_OF_DATA_LEARN_FILE 
    counter = 1
    evolutionOfFail = []
    countFail = printNumberSuccessAndFail(listError)
    evolutionOfFail.append(countFail)
    
    error = NUMBER_OF_DATA_LEARN_FILE
    evolutionOfError = []
    while  error>EPSILON_ERROR:
    
        for i in range(0,NUMBER_OF_DATA_LEARN_FILE): #Apprend sur tous le set
            data = init_data(i)
            neurons = calculatePropagation(data,neurons)
            neurons = ApplicateCorrectionToWeight(data,neurons) # On applique la correction des poids 

        error= 0
        for i in range(0,NUMBER_OF_DATA_LEARN_FILE): #Calcul les erreurs restantes sans modifier les poids
            data = init_data(i)
            neurons = calculatePropagation(data,neurons)
            resCalculerOfficial = calculateOfficialResult(neurons)
            listError[i]= ( resCalculerOfficial==data.officialResult)
            error += CalculateErrorTotal(data,neurons)
            countFail = printNumberSuccessAndFail(listError)
            evolutionOfFail.append(countFail)
        evolutionOfError.append(error)
        counter +=1
    return neurons,evolutionOfError


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
         # on bruite le data set
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


def initWeight(neuron,nombrePoids) :
    
    neuron.weightLayer = []
    for i in range(nombrePoids):
        neuron.weightLayer.append(rand.random()/(nombrePoids*10))



def init_neuron(nombrePoids):
    neurons = []
    NombreNeuronSortie = NUMBERS_TO_DETECT
    for i in range(NombreNeuronSortie):
        newNeuron = neuron_layer()
        initWeight(newNeuron,nombrePoids)
        neurons.append(newNeuron)
    return neurons
    
        
#permet d'initialiser en chargeant un premier fichier et d'initialiser un certain nombre de neurones
def init():
    data = init_data(0)
    nbPoids = len(data.input)
    return init_neuron(nbPoids)


def init_neuron_test(nombrePoids):
    neurons = []
    NombreNeuronSortie = NUMBERS_TO_DETECT
    for i in range(NombreNeuronSortie):
        newNeuron = neuron_layer()
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
    counter = 1
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

neurons = init()
neurons,evolutionOfError = runDataLearnMethod2(neurons)
runAllFileToTest(neurons)


title = "Evolution de l'erreur avec epsilon="+str(EPSILON_ERROR)
PrintManagerEvolutionOfError(evolutionOfError,title)
courbeGeneralisation(neurons)

print("FIN")


pylab.show()

