from sre_constants import SUCCESS
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

weightLayer = []
SEUIL_ACTIVATION = 0.5
TAUX_APPRENTISSAGE = 0.01
dataFileIndex = 0
NUMBER_OF_DATA_LEARN_FILE = 2
EPSILON_ERROR = 0.001

def init_data(dataFileIndex,learn=True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if(learn):
        dataFile = open(dir_path+"/data_learn/data"+str(dataFileIndex)+".txt", "r")
    else :
        dataFile = open(dir_path+"/data_bruit/data"+str(dataFileIndex)+".txt", "r")
    
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
    return data


#permet d'initaliser des poids de facon aléaoitre
def init_weight(data):
    n = len(data.input)
    for i in range(0,n):
        weightLayer.append(rand.random()/n)
    

#Effectue une propagation, en calculant avec les datas/poids actuels
#Retourne le resultat de la propagation
def calculatePropagation(data):
    res = 0.0
    length =len(data.input)
    for i in range(0,length):
        res += data.input[i]*weightLayer[i]
    if(res<SEUIL_ACTIVATION):
        return res,0
    else:
        return res,1

#Retourne le résultat de la sigmoid sur l'argument d'entré
def sigmoid(result ):
    result = 1 / (1 + math.exp(-result))
    return result

def ApplicateCorrectionToWeight(data,res):
    erreurCalcule =   data.officialResult-res
    for i in range(0,len(weightLayer)):
        weightLayer[i] +=TAUX_APPRENTISSAGE*erreurCalcule*data.input[i]


#Return true if all result of data set are good
# if one or more fail, return False
def AreAllTestSuccess(list):
    for i in range(0,NUMBER_OF_DATA_LEARN_FILE):
        if(list[i]==False):
            return False
    return True


#Retourne l'évolution de l'erreur suivant le nombre d'itération
def runDataLearnMethod2():
    listError = [False]*NUMBER_OF_DATA_LEARN_FILE 
    counter = 1
    evolutionOfFail = []
    countFail = printNumberSuccessAndFail(listError)   
    error = NUMBER_OF_DATA_LEARN_FILE
    evolutionOfError = []
    evolutionOfError.append(error)

    #On continue de boucler tant que tout le data set n'est pas valide 
    # while  AreAllTestSuccess(listError)==False or countOfSuccessInARow<NUMBER_OF_DATA_LEARN_FILE:
    while  error>EPSILON_ERROR:
    # for n in range(0,100):
        error = 0
        for i in range(0,NUMBER_OF_DATA_LEARN_FILE):
            data = init_data(i)
            analogicAnswer,officialRes = calculatePropagation(data)
            listError[i]= (officialRes==data.officialResult)            
            error += abs(data.officialResult-analogicAnswer)
            ApplicateCorrectionToWeight(data,analogicAnswer) # On applique la correction des poids 
            counter +=1
            countFail = printNumberSuccessAndFail(listError)
            evolutionOfFail.append(countFail)
        evolutionOfError.append(error)
    return evolutionOfError

#permet de faire apprendre au réseau en utilisant les données binaires (méthode 1)
def runDataLearnMethod1():
    
    listError = [False]*NUMBER_OF_DATA_LEARN_FILE 
    counter = 1
    evolutionOfFail = []
    countFail = printNumberSuccessAndFail(listError)
    evolutionOfFail.append(countFail)
    
    countOfSuccessInARow = 0

    evolutionOfError = []
    evolutionOfError.append(2)
    error = 2
    #On continue de boucler tant que tout le data set n'est pas valide 
    while  AreAllTestSuccess(listError)==False or countOfSuccessInARow<NUMBER_OF_DATA_LEARN_FILE:
        error = 0
        for i in range(0,NUMBER_OF_DATA_LEARN_FILE):
            data = init_data(i)
            analogicAnswer,officialRes = calculatePropagation(data)
            listError[i]= (officialRes==data.officialResult)
            if(AreAllTestSuccess(listError)):
                countOfSuccessInARow +=1
            else:
                countOfSuccessInARow =0

            ApplicateCorrectionToWeight(data,officialRes) # On applique la correction des poids 
            counter +=1
            countFail = printNumberSuccessAndFail(listError)
            evolutionOfFail.append(countFail)
            error += abs(data.officialResult-officialRes)
        evolutionOfError.append(error)
    return evolutionOfError

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
def run50timesOnDataModified(data,tauxBruit):
    nbIteration = 50
    listError = [False]*nbIteration 
    counter = 1
    for i in range(0,nbIteration):
        dataModified = bruitage(data,tauxBruit)
        
        res,officialRes = calculatePropagation(dataModified)
        listError[i]= (officialRes==dataModified.officialResult)
        counter +=1
    counterOfError = printNumberSuccessAndFail(listError)
    return counterOfError

def courbeGeneralisationMotif0():
    evolutionOfFail = []
    data = init_data(0)
    for i in range(0,100):
        counterOfFail = run50timesOnDataModified(data,i)
        evolutionOfFail.append(counterOfFail*2)
    return evolutionOfFail

def courbeGeneralisationMotif1():
    evolutionOfFail = []
    data = init_data(1)
    for i in range(0,100):
        counterOfFail = run50timesOnDataModified(data,i)
        evolutionOfFail.append(counterOfFail*2)
        
    return evolutionOfFail
def courbeGeneralisation():
    evolutionOfFail0 = courbeGeneralisationMotif0()
    evolutionOfFail1 = courbeGeneralisationMotif1()
    return evolutionOfFail0,evolutionOfFail1


 


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

#permet d'initialiser en chargeant un premier fichier et d'initialiser le nombre de poids en fonction
def init():
    data = init_data(0)
    init_weight(data)


def PrintManagerEvolutionOfFail(evolutionOfFail0,evolutionOfFail1):
    plt.plot(evolutionOfFail0,label="Courbe de generalisation Motif 0")
    plt.plot(evolutionOfFail1,label="Courbe de generalisation Motif 1")
    plt.xlabel("Taux de bruitage")
    plt.ylabel("Pourcentage erreur")
    plt.legend()
    pylab.show()

def PrintManagerEvolutionOfError(evolutionOfError,title):
    plt.plot(evolutionOfError,label=title)
    plt.xlabel("Nombre itérations")
    plt.ylabel("Erreur")
    
    plt.legend()
    pylab.show()

init()
evolutionOfError = runDataLearnMethod1()
PrintManagerEvolutionOfError(evolutionOfError,"Evolution de l'erreur - Perceptron simple")
evolution0,evolution1 = courbeGeneralisation()
PrintManagerEvolutionOfFail(evolution0,evolution1)


weightLayer.clear()
init()
evolutionOfError = runDataLearnMethod2()


PrintManagerEvolutionOfError(evolutionOfError,("Evolution de l'erreur avec Epsilon = 0.001"))
courbeGeneralisationMotif0()
evolution0,evolution1 = courbeGeneralisation()
PrintManagerEvolutionOfFail(evolution0,evolution1)





