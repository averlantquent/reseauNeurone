# reseauNeurone
TD sur la création d'un réseau de neurones

Le fichier perceptronSimple comprend le code pour la partie 1 et la partie 2.
Il utilise le dossier data_learn comme dataset. Il Contient les motifs pour un 0 et un 1.
Pour la partie trois, il faut lancer le fichier separation10classes. Il va utiliser le dossier data_learn_10 qui contient les motifs entre 0 et 9. 
Pour la partie 4, deux codes sont disponibles :  
    deepLearning.py : Utilise comme fichier source les 10 motifs précédemment créés.
    deepLEarningMnist.py : Utilise la base Mnist pour faire l'apprentissage et les tests. 
Pour lancer ce deepLEarningMnist, il faut d'abord installer keras ainsi que tensorflow. (pip install keras, pip install tensoflow).4
Afin de limiter le temps pour la phase d'apprentissage, nous avons réduit le nombre d'itération pour la phase d'apprentissage en passant de 100000 à 10000. Libre à vous de changer cette valeur.