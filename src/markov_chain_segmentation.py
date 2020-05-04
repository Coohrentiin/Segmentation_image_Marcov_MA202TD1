#TP1: CMC

# Segmentation Bayésienne
# Chaines de Markov

#%% Importation des librairies: 
import numpy as np
from tools import bruit_gauss, calc_erreur
from markov_chain import *

#%% Paramètres de la simulation:
n=1000
w=np.array([0,255])
p=np.array([0.25,0.75])
A=np.array( [ [0.8, 0.2],
			  [0.07, 0.93]])
m1=0
sig1=1
m2=3
sig2=2

#%% Traitement du signal
### Génération du signal:
signal=simu_mc(n,w,p,A)
### Génération du bruit:
signal_noisy= bruit_gauss(signal,w,m1,sig1,m2,sig2)
### Optention du nouveau signal par le mode des maginales à postériori:
new_signal=mpm_mc(signal_noisy,w,p,A,m1,sig1,m2,sig2)
### Calcul de l'erreur:
erreur=calc_erreur(signal,new_signal)
print("Erreur commise: {:.2%}".format(erreur))

# Application à la segmentation d'image: 
# %%
