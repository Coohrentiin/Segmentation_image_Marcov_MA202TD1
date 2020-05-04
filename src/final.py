# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TP1: CMC
# 
# ## 1. Segmentation Bayésienne
# 
# L'objectif de ce TD est de réaliser une segmentation bayésienne de données. On cherche donc à classifier $n$ réalisation alétoire observés par une variable $Y$ en une ralisation de l'ensemble des classes $\Omega$. Ainsi on cherche à estimer $X=(X_1, ... X_n)$ à partir de l'observation $Y=(Y_1, ... Y_n)$ tout en minimisant la perte, correspondant à une erreur, modélisé par la fonction: 
# $$\forall(i,j)\in\Omega^2, L_{0/1}(i,j)=\mathbb{1}_{[i\neq j]}$$
# Pour minimiser cette fonction de perte on peut utiliser un estimateur MPM (mode des marginales à postériori), le problèmes se limite alors à la modélisation de la loi $p(x,y)$ régissant le lien entre la segementation $X$ et l'observation $Y$. Nous comparerons ici une modélisation de chaine de Markov introduisant une "dépendance d'une donnée à son passé", les données $(X_i,Y_i)$ sont alors dépendantes. Et le modèles des $(X_i,Y_i)$ indépendants, prenant chaque élement indépandament aux autres.
# 
# ## 2. Chaines de Markov
# 
# Dans ce cadre nous modélisons un processus aléatoire de $n$ réalisations pouvant siéger dans deux classes différentes $\Omega={\omega_1,\omega_2}$, avec $X$ un processus caché $X=(X_1, ... X_n)\in\Omega$ que l'on estime à partir de l'observation $Y=(Y_1, ... Y_n)\in\mathbb{R}$. X étant une chaine de Markov caché ( $(X,Y)$ de Markov).
# 
# Importation des librairies:

# %%
import os
path=os.getcwd()+"/src" 
os.chdir(path)
print(os.getcwd())

import numpy as np
from tools import bruit_gauss, calc_erreur
from markov_chain import *

np.random.seed(1) #1 totalement arbitraire, je fixe la graine aléatoire de manière à rendre mes observations consitantes aux données. 

# %% [markdown]
# Paramètres de la simulation:
# 
# Pour la segmentation d'image nous considérons les deux états noir et blanc corrospondant aux nombres 255 et 0 (codage sur 8 bits).
# 
# 
# %% [markdown]
# Cf.README figure 1
# ```mermaid
# graph LR;
#     0-.0,2.->255;
#     0-.0,8.->0;
#     255-.0,07.->0;
#     255-.0,93.->255;
# ```
# 

# %%
n=1000

w=np.array([0,255])
p=np.array([0.25,0.75])
A=np.array( [ [0.8, 0.2],
			  [0.07, 0.93]])
#Configuration n°1:
m1=0
m2=3
sig1=1
sig2=2
config=[[m1,m2,sig1,sig2]]
#Configuration n°2:
m1=1
m2=1
sig1=1
sig2=5
config.append([m1,m2,sig1,sig2])
#Configuration n°3:
m1=0
m2=1
sig1=1
sig2=1
config.append([m1,m2,sig1,sig2])

# %% [markdown]
# ### Modèle des chaines de Markov:
# Création du signal et son bruitage, reconstruction du signal avec le mode des marginales à postériori:

# %%
### Génération du signal:
signal=simu_mc(n,w,p,A)

### Génération du bruit avec les différentes configurations:
signal_noisy=[]
for conf in config:
    signal_noisy.append( bruit_gauss(signal,w,conf[0],conf[2],conf[1],conf[3]) )

### Optention du nouveau signal par le mode des maginales à postériori:
new_signal=[]
i=0
for conf in config:
    new_signal.append(mpm_mc(signal_noisy[i],w,p,A,conf[0],conf[2],conf[1],conf[3]))
    i=i+1

### Calcul de l'erreur:
erreur=[]
i=0
for conf in config:
    erreur.append(calc_erreur(signal,new_signal[i]))
    print("Erreur commise avec la configuration n°{}: {:.2%}".format(i,erreur[i]))
    i=i+1

# %% [markdown]
# Dans l'hypothèse de stationarité, et en utilisant l'estimateur empirique des fréquences, on peut à partir du signal (non bruité) donner une estimation de la répatition de départ $p$ et de la matrice de transition $A$. 

# %%

p_estim,A_estim=calc_probaprio_mc(signal,w)
print('p estimé:')
print(p_estim)
print('p reel:')
print(p)
print('A estimé:')
print(A_estim)
print('A reel:')
print(A)

# %% [markdown]
# ### Modèle idépendant: 
# 
# On va maintenant comparer le résultat de l'estimation offerte par Markov a l'estimation que l'on peut avoir avec un modèle idépendant. 
# 
# #### génération du signal en modèle indépendant: 

# %%
from gaussian_mixture import *
from matplotlib import pyplot as plt
n=1000
np.random.seed(1)
w=np.array([0,255])
p=np.array([0.25,0.75])
A=np.array( [ [0.8, 0.2],
			  [0.07, 0.93]])

##*********************************
## Modèle indépendant:
##*********************************
### Génération du signal:
signal=simu_gm(n,w,p)

### Génération du bruit avec les différentes configurations:
signal_noisy=[]
for conf in config:
    signal_noisy.append( bruit_gauss(signal,w,conf[0],conf[2],conf[1],conf[3]) )

### Optention des paramètres d'une chaine de Markov
p_op,A_op=calc_probaprio_mc(signal, w)

### Optention du nouveau signal par le mode des maginales à postériori pour le modèle indep et de markov:
new_signal_indep=[]
new_signal_markov=[]
i=0
for conf in config:
    new_signal_indep.append(mpm_gm(signal_noisy[i],w,p,conf[0],conf[2],conf[1],conf[3]))
    new_signal_markov.append(mpm_mc(signal_noisy[i],w,p_op,A_op,conf[0],conf[2],conf[1],conf[3]))
    i=i+1

### Calcul de l'erreur:
erreur_indep=[]
erreur_markov=[]
i=0
for conf in config:
    erreur_indep.append(calc_erreur(signal,new_signal_indep[i]))
    erreur_markov.append(calc_erreur(signal,new_signal_markov[i]))
    print("Erreur commise avec la configuration n°{}: suivant le modèle indépendant:{:.2%} , suivant le modèle de Markov {:.2%}".format(i,erreur_indep[i],erreur_markov[i]))
    i=i+1

plt.plot([1,2,3],erreur_indep,label="erreurs modèle indépendant")
plt.plot([1,2,3],erreur_markov,label="erreurs modèle de Markov")
plt.title("Comparaison des erreurs dans le cas d'un modèle de signal indépendant")
plt.legend()
plt.show()
print(A_op)
print(p_op)

# %% [markdown]
# Nous remarquons que les erreurs du modèle de Markov sont légèrement meilleures que celles du modèle indépendant néamoins en faisant varier la racine aléatoire, on peut remarquer que même si globalement Markov reste meilleurs, il peut parfois être moins bon. 
# 
# #### génération du signal en modèle de Markov:
# 

# %%
##*********************************
## Modèle de Markov:
##*********************************
### Génération du signal:
signal=simu_mc(n,w,p,A)

### Génération du bruit avec les différentes configurations:
signal_noisy=[]
for conf in config:
    signal_noisy.append( bruit_gauss(signal,w,conf[0],conf[2],conf[1],conf[3]) )

### Optention des paramètres du modèle indep
p_indep_op=calc_probaprio_gm(signal, w)

### Optention du nouveau signal par le mode des maginales à postériori pour le modèle indep et de markov:
new_signal_indep=[]
new_signal_markov=[]
i=0
for conf in config:
    new_signal_indep.append(mpm_gm(signal_noisy[i],w,p_indep_op,conf[0],conf[2],conf[1],conf[3]))
    new_signal_markov.append(mpm_mc(signal_noisy[i],w,p,A,conf[0],conf[2],conf[1],conf[3]))
    i=i+1

### Calcul de l'erreur:
erreur_indep=[]
erreur_markov=[]
i=0
for conf in config:
    erreur_indep.append(calc_erreur(signal,new_signal_indep[i]))
    erreur_markov.append(calc_erreur(signal,new_signal_markov[i]))
    print("Erreur commise avec la configuration n°{}: suivant le modèle indépendant:{:.2%} , suivant le modèle de Markov {:.2%}".format(i,erreur_indep[i],erreur_markov[i]))
    i=i+1

plt.plot([1,2,3],erreur_indep,label="erreurs modèle indépendant")
plt.plot([1,2,3],erreur_markov,label="erreurs modèle de Markov")
plt.title("Comparaison des erreurs dans le cas d'un modèle de signal de Markov")
plt.legend()
plt.show()
print(np.sum(np.array(erreur_indep)/np.array(erreur_markov))/3)

# %% [markdown]
# On remarque cette fois ci que dans le cas d'un signal généré par un modèle de Markov, l'estimation est bien meilleurs en utilisant l'estimation avec Markov qu'avec le modèle indépendant.
# Cela prend sens dans la mesure que si les données sont générés avec un dépendance le modèle de Markov va l'interpréter lors de la restitution alors que le modèle indépent passe au travers.
# 
# ### Conclusion:
# On remarque que l'on ne perd à priori pas notre temps à utiliser des Modèles de Markov, en effet dans le cas de signaux indepands, l'estimation de MPM via modèle de Markov n'entraine pas significativement plus d'erreur que le modèle indépendant, alors que l'inverse est fausse, l'estimation indépendante entraine $\sim2$ fois plus d'erreur en moyenne que le modèle de Markov
# 
# ### Approfondissement:
# Dans les résultats précedent nous travaillons sur des obsevations qui ne sont pas consistante dans la mesure où comme on l'a fait remarquer précedament les résultats dépendent de la racine aléatoire. Je propose ainsi d'utiliser la loi des grand nombre pour justifier ma conclusion. Je realise ainsi les mêmes calculs que précedement mais pour 1000 racines aléatoire différentes. Puis je représente $\frac{erreur\ avec\ MPM\ indépendant}{erreur\ avec\ MPM\ de\ Markov}$ pour le signal modèle indépendant et modèle de Markov sous la forme d'une histogramme de fréqunece en classant le rapport d'erreur en 100 classes. On remarque alors que d'une par les distributions semblent bien être distribution gaussienne (ce qui valide ma tentative d'aller vers un LDGN), que ces gaussienne ne semblent pas se supperposer et que l'une semble centré en 1 tendis que l'autre semble centrée en 2. Cela correpond bien à ma conclusion précédente. Cette observation pourrait être justifié plus "mathématiquement" par un test de recouvrement de modèle mais c'est pas le sujet ici. 

# %%
e_indep=[]
e_markov=[]
n=1000
for seed in range(200):
    np.random.seed(seed)
    ##*********************************
    ## Modèle indépendant:
    ##*********************************
    ### Génération du signal:
    signal=simu_gm(n,w,p)

    ### Génération du bruit avec les différentes configurations:
    signal_noisy=[]
    for conf in config:
        signal_noisy.append( bruit_gauss(signal,w,conf[0],conf[2],conf[1],conf[3]) )

    ### Optention des paramètres d'une chaine de Markov
    p_op,A_op=calc_probaprio_mc(signal, w)

    ### Optention du nouveau signal par le mode des maginales à postériori pour le modèle indep et de markov:
    new_signal_indep=[]
    new_signal_markov=[]
    i=0
    for conf in config:
        new_signal_indep.append(mpm_gm(signal_noisy[i],w,p,conf[0],conf[2],conf[1],conf[3]))
        new_signal_markov.append(mpm_mc(signal_noisy[i],w,p_op,A_op,conf[0],conf[2],conf[1],conf[3]))
        i=i+1

    ### Calcul de l'erreur:
    erreur_indep=[]
    erreur_markov=[]
    i=0
    for conf in config:
        erreur_indep.append(calc_erreur(signal,new_signal_indep[i]))
        erreur_markov.append(calc_erreur(signal,new_signal_markov[i]))
        i=i+1

    e_indep.append(np.sum(np.array(erreur_indep)/np.array(erreur_markov))/3)
    ##*********************************
    ## Modèle de Markov:
    ##*********************************
    ### Génération du signal:
    signal=simu_mc(n,w,p,A)

    ### Génération du bruit avec les différentes configurations:
    signal_noisy=[]
    for conf in config:
        signal_noisy.append( bruit_gauss(signal,w,conf[0],conf[2],conf[1],conf[3]) )

    ### Optention des paramètres du modèle indep
    p_indep_op=calc_probaprio_gm(signal, w)

    ### Optention du nouveau signal par le mode des maginales à postériori pour le modèle indep et de markov:
    new_signal_indep=[]
    new_signal_markov=[]
    i=0
    for conf in config:
        new_signal_indep.append(mpm_gm(signal_noisy[i],w,p_indep_op,conf[0],conf[2],conf[1],conf[3]))
        new_signal_markov.append(mpm_mc(signal_noisy[i],w,p,A,conf[0],conf[2],conf[1],conf[3]))
        i=i+1

    ### Calcul de l'erreur:
    erreur_indep=[]
    erreur_markov=[]
    i=0
    for conf in config:
        erreur_indep.append(calc_erreur(signal,new_signal_indep[i]))
        erreur_markov.append(calc_erreur(signal,new_signal_markov[i]))
        i=i+1
    e_markov.append(np.sum(np.array(erreur_indep)/np.array(erreur_markov))/3)

e_indep=np.array(e_indep)
e_markov=np.array(e_markov)

plt.hist(e_indep,bins=100,color="red",alpha=0.5,label="modèle indep")
plt.hist(e_markov,bins=100,color="blue",alpha=0.5,label="modèle markov")
plt.title("Histogramme de la fréquence des rapports d'erreur")
plt.legend()
plt.show()

# %% [markdown]
# # 3. Application à la segmentation d'images

# %%
import numpy as np
import cv2 as cv
from tools import bruit_gauss,calc_erreur, peano_transform_img, transform_peano_in_img, line_transform_img, transform_line_in_img
from gaussian_mixture import *
from markov_chain import *
from matplotlib import pyplot as plt

img=cv.imread("../images/zebre2.bmp",cv.IMREAD_GRAYSCALE )
plt.imshow(img, cmap='gray')
plt.show()
print(img.shape)
print(img.size)
signal_image=peano_transform_img(img)
print(signal_image.shape)

# %% [markdown]
# Bruitons l'image obtenus avec une bruit gaussiens qui sera inconue dans la suite du raisonnement:
# 

# %%
m1=1
sig1=4
m2=3
sig2=2
w=np.array([0,255])
signal_noisy_image=bruit_gauss(signal_image,w,m1,sig1,m2,sig2)
img_noisy=transform_peano_in_img(signal_noisy_image,256)
plt.imshow(img_noisy, cmap='gray')
plt.show()

# %% [markdown]
# A partir de ce signal bruité, l'ogjectif est d'estimer les paramètres de la loi de la chaine de Markov et du bruit, afin de restaurer le signal.

# %%
(p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)=estim_param_EM_mc(10,signal_noisy_image,p,A,m1,sig1,m2,sig2)
new_signal_image=mpm_mc(signal_noisy_image,w,p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)


# %%
img_noisy=transform_peano_in_img(new_signal_image,256)
plt.imshow(img_noisy, cmap='gray')
plt.show()
erreur=calc_erreur(signal_image,new_signal_image)
print("Erreur commise: {:.2%}".format(erreur))


# %%


