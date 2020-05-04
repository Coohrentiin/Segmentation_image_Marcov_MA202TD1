#test pour verifier que le compilateur fonctionne: 

#%% Librairies
import tools
import numpy as np
import markov_chain as mc

#%% test fonction 1:
signal=np.array([1,2,1,1,2,1])
w=[1,2]

m1=0
m2=3
sig1=2
sig2=1

signal_noisy=tools.bruit_gauss(signal, w, m1, sig1, m2, sig2)
print(signal_noisy)

# %% test fonction 2
#permet de connaitre telle ou telle gaussienne 

pb=tools.gauss(signal_noisy,m1,sig1,m2,sig2)
print(pb)
# %%
pb[:,0]