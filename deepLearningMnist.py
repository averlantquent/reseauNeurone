from itertools import count
from sre_constants import SUCCESS
import time
import numpy as np
import random as rand
import os
import math
import matplotlib.pyplot as plt
import pylab
from keras.datasets import mnist
from contextlib import contextmanager

NUMBERS_TO_DETECT = 10
SEUIL_ACTIVATION = 0.5
TAUX_APPRENTISSAGE = 1
NUMBER_OF_DATA_LEARN_FILE = 59000
EPSILON_ERROR = 0.1
NUMBER_OF_FILE_TO_TEST = 20000



@contextmanager
def timeit_context(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print('[{}] finished in {} ms'.format(name, int(elapsed_time * 1_000)))

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







def init_data(train_X,train_Y,numberSelected=0):
    
    
    train_XChoose = train_X[numberSelected]
    data = dataLearn()
    originalListe = []
    for array in train_XChoose:
        
        liste = array.tolist()
        originalListe.extend(liste)
    for value in originalListe:
        data.input.append(value/255)
        
    data.officialResult = train_Y[numberSelected]
    data.arrayOfficialResult[data.officialResult] =1 #On met un 1 pour la case d'arrivée
    # print("official result = "+str(data.officialResult))
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


def runDataLearnMethod2(firstLayer,lastLayer,train_X, train_y):
    listError = [False]*NUMBER_OF_DATA_LEARN_FILE 
    counter = 1
    evolutionOfFail = []
    countFail = printNumberSuccessAndFail(listError)
    # evolutionOfFail.append(countFail)
    
    error = NUMBER_OF_DATA_LEARN_FILE
    evolutionOfError = []
    while  error>EPSILON_ERROR and counter<2000: #on fixe une limite en cas de boucle trop longue
        
        # for i in range(0,NUMBER_OF_DATA_LEARN_FILE): #Apprend sur tout le set
        rankMotifSelected = rand.randint(0, 59000)
        data = init_data(train_X, train_y,rankMotifSelected)
        firstLayer,lastLayer = calculatePropagation(data,firstLayer,lastLayer)
        firstLayer,lastLayer = ApplicateCorrectionToWeight(data,firstLayer,lastLayer) # On applique la correction des poids 
            
        # error= 0
        # for i in range(0,NUMBER_OF_DATA_LEARN_FILE): #Calcul les erreurs restantes sans modifier les poids
        #     data = init_data(train_X, train_y,i)
        #     firstLayer,lastLayer = calculatePropagation(data,firstLayer,lastLayer)
        #     listError[i]= ( calculateOfficialResult(lastLayer)==data.officialResult)
        #     error += CalculateErrorTotal(data,lastLayer)
        #     # print("Résultat trouvé : ",calculateOfficialResult(lastLayer)," Resultat officiel : ",data.officialResult)
            
        #     countFail = printNumberSuccessAndFail(listError)
        #     evolutionOfFail.append(countFail)

        # print("Counter :",counter," error : ",error)
        print("Counter :",counter)
        # evolutionOfError.append(error)
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





def plotCout(listeCout):
    plt.plot(listeCout)   



        
#permet d'initialiser en chargeant un premier fichier et d'initialiser un certain nombre de neurones
def init(nbPoids):
    # data = init_data(0)
    # nbPoids = len(data.input)
    return init_neurons(nbPoids)



#Permet de sauvergarder l'ensemble des poids dans un fichier pour une lecture future
def writeNeuronWeightOnFile(neurons,index):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFile = open(dir_path+"/save_neurons/save"+str(index)+".txt", "w")

    for neuron in neurons:
        dataFile.write(str(neuron.weightLayer))    
        dataFile.write("\n")
    dataFile.close()
    




def PrintManagerEvolutionOfError(evolutionOfError,title):
    plt.plot(evolutionOfError,label=title)
    plt.xlabel("Nombre itérations")
    plt.ylabel("Erreur")    
    plt.legend()
    pylab.show()

def runDataTest(firstLayer,lastLayer,test_X, test_y):
    listError = [False]*NUMBER_OF_FILE_TO_TEST 
    counter = 1
    evolutionOfFail = []
    countFail = printNumberSuccessAndFail(listError)
    evolutionOfFail.append(countFail)
    
    error = NUMBER_OF_DATA_LEARN_FILE
    evolutionOfError = []
    
    error= 0
    for i in range(0,NUMBER_OF_FILE_TO_TEST-1): #Calcul les erreurs restantes sans modifier les poids
        data = init_data(test_X, test_y,i)
        firstLayer,lastLayer = calculatePropagation(data,firstLayer,lastLayer)
        listError[i]= ( calculateOfficialResult(lastLayer)==data.officialResult)
        # error += CalculateErrorTotal(data,lastLayer)
        print("Résultat trouvé : ",calculateOfficialResult(lastLayer)," Resultat officiel : ",data.officialResult)
        

        # evolutionOfFail.append(countFail)

        # print("Counter :",counter," error : ",error)
        # evolutionOfError.append(error)
        counter +=1
    countFail = printNumberSuccessAndFail(listError)
    print("nombre iteration : ",counter)
    return countFail

(train_X, train_y), (test_X, test_y) = mnist.load_data()

data = init_data(train_X,train_y)

nombreNeuronCouche1 = 100
nombrePoidsParNeuronCouche1 = len(data.input)
neuronsFirstLayer = init_neurons(nombrePoidsParNeuronCouche1,nombreNeuronCouche1)

nombreNeuronCouche2 = NUMBERS_TO_DETECT
nombrePoidsParCouche2 = nombreNeuronCouche1
neuronsSecondLayer = init_neurons(nombrePoidsParCouche2,nombreNeuronCouche2)

# data = init_data(10)
# firstLayer,lastLayer = calculatePropagation(data,neuronsFirstLayer,neuronsSecondLayer)

neuronsFirstLayer,neuronsSecondLayer,evolutionOfError = runDataLearnMethod2(neuronsFirstLayer,neuronsSecondLayer,train_X, train_y)
# # writeNeuronWeightOnFile(neurons,0)
#runAllFileToTest(neurons)
# # runTest()

title = "Evolution de l'erreur avec epsilon="+str(EPSILON_ERROR)
PrintManagerEvolutionOfError(evolutionOfError,title)
# courbeGeneralisation(neurons)
countFail = runDataTest(neuronsFirstLayer,neuronsSecondLayer,test_X, test_y)
print("Count fail ",countFail)
print("FIN")
# init()
# runDataLearn()

# courbeGeneralisation()

pylab.show()

