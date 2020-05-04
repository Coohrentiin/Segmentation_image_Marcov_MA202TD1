import numpy as np
from markov_chain import *
import cv2 as cv
from tools import bruit_gauss,calc_erreur, peano_transform_img, transform_peano_in_img, line_transform_img, transform_line_in_img
from gaussian_mixture import *
from matplotlib import pyplot as plt

def ttm(name):
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

	path="../images/%s" % name
	img=cv.imread(path,cv.IMREAD_GRAYSCALE )
	signal_image=peano_transform_img(img)

	w=np.array([0,255])
	img_noisy=[]
	new_img=[]
	new_img_indep=[]
	erreur=[]
	erreur_indep=[]
	for conf in config:
		signal_noisy_image=bruit_gauss(signal_image,w,conf[0],conf[2],conf[1],conf[3])
		img_noisy.append(transform_peano_in_img(signal_noisy_image,256))
		(p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)=tt_estim_param_EM_mc(10,signal_noisy_image,p,A,conf[0],conf[2],conf[1],conf[3])
		new_signal_image=mpm_mc(signal_noisy_image,w,p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)
		new_img.append(transform_peano_in_img(new_signal_image,256))
		erreur.append(calc_erreur(signal_image,new_signal_image))
		(p_est, m1_est, sig1_est, m2_est, sig2_est)=tt_estim_param_EM_gm(10,signal_noisy_image,p,conf[0],conf[2],conf[1],conf[3])
		new_signal_image=mpm_gm(signal_noisy_image,w,p_est, m1_est, sig1_est, m2_est, sig2_est)
		new_img_indep.append(transform_peano_in_img(new_signal_image,256))
		erreur_indep.append(calc_erreur(signal_image,new_signal_image))
	plt.figure(figsize=(9, 3))
	plt.imshow(img, cmap='gray')
	plt.title("Image d'origine:")
	plt.show()

	plt.figure(figsize=(9, 3))
	plt.subplot(131)
	plt.imshow(img_noisy[0], cmap='gray')
	plt.title('mu=({},{}) sig=({},{})'.format(config[0][0],config[0][1],config[0][2],config[0][3]))
	plt.subplot(132)
	plt.imshow(img_noisy[1], cmap='gray')
	plt.title('mu=({},{}) sig=({},{})'.format(config[1][0],config[1][1],config[1][2],config[1][3]))
	plt.subplot(133)
	plt.imshow(img_noisy[2], cmap='gray')
	plt.title('mu=({},{}) sig=({},{})'.format(config[2][0],config[2][1],config[2][2],config[2][3]))
	plt.suptitle("Image bruité, en fonction de la configuration du bruit:")
	plt.show()

	plt.figure(figsize=(9, 3))
	plt.subplot(131)
	plt.imshow(new_img[0], cmap='gray',label=erreur[0])
	plt.title('Erreur:{:.2%}'.format(erreur[0]))
	plt.subplot(132)
	plt.imshow(new_img[1], cmap='gray',label=erreur[1])
	plt.title('Erreur:{:.2%}'.format(erreur[1]))
	plt.subplot(133)
	plt.imshow(new_img[2], cmap='gray',label=erreur[2])
	plt.title('Erreur:{:.2%}'.format(erreur[2]))
	plt.suptitle("Image restitué, en fonction de la configuration du bruit, pour le modèle de Markov:")
	plt.show()

	plt.figure(figsize=(9, 3))
	plt.subplot(131)
	plt.imshow(new_img_indep[0], cmap='gray',label=erreur_indep[0])
	plt.title('Erreur:{:.2%}'.format(erreur_indep[0]))
	plt.subplot(132)
	plt.imshow(new_img_indep[1], cmap='gray',label=erreur_indep[1])
	plt.title('Erreur:{:.2%}'.format(erreur_indep[1]))
	plt.subplot(133)
	plt.imshow(new_img_indep[2], cmap='gray',label=erreur_indep[2])
	plt.title('Erreur:{:.2%}'.format(erreur_indep[2]))
	plt.suptitle("Image restitué, en fonction de la configuration du bruit, pour le modèle indépendant:")
	plt.show()

## exactement les même fonctions que markov chain mais sans les prints 
def tt_calc_param_EM_mc(signal_noisy, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction permet de calculer les nouveaux paramètres estimé pour une itération de EM
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: tous les paramètres réestimés donc p, A, m1, sig1, m2, sig2
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    alpha = forward(A, p, gausses)
    beta = backward(A, gausses)
    # print("alpha*beta={}".format(alpha*beta))

    #calcul de la loi marginale à postériori p(xn|y)
    a_posteriori=( alpha*beta) / (alpha*beta).sum(axis=1)[...,np.newaxis] #loi marginale à postériori p(xn|y)
    # print("pb à postério:{}".format(a_posteriori))

    #pi_k, probabilités de se trouver dans un état k
    p=a_posteriori.sum(axis=0)/a_posteriori.shape[0]
    # print("p={}".format(p))

    # calcul de p^k_{ij,n} 
    c_a_posteriori= ( alpha[:-1, :, np.newaxis]
                    * (gausses[1:, np.newaxis, :]
                    * beta[1:, np.newaxis, :]
                    * A[np.newaxis,:,:]) )   
    c_a_posteriori=c_a_posteriori/ (c_a_posteriori.sum(axis=(1,2))[...,np.newaxis, np.newaxis])  

    #Estimation de la matrice de transition (hypothèse: stationarité)
    A= np.transpose( np.transpose( (c_a_posteriori.sum(axis=0))) / (a_posteriori[:-1:].sum(axis=0))) 

    #Estimation de la moyenne et la variance
    m1= (a_posteriori[:,0]*signal_noisy).sum()/a_posteriori[:,0].sum()
    m2= (a_posteriori[:,1]*signal_noisy).sum()/a_posteriori[:,1].sum()
    sig1= np.sqrt( (a_posteriori[:,0]*((signal_noisy-m1)**2)).sum()/a_posteriori[:,0].sum())
    sig2= np.sqrt( (a_posteriori[:,1]*((signal_noisy-m2)**2)).sum()/a_posteriori[:,1].sum())
    return p,A,m1,sig1,m2,sig2


def tt_estim_param_EM_mc(iter, signal_noisy, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction est l'implémentation de l'algorithme EM pour le modèle en question
    :param iter: Nombre d'itération choisie
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: la valeur d'initialisation du vecteur de proba
    :param A: la valeur d'initialisation de la matrice de transition de la chaîne
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de la variance de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de la variance de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme EM donc p, A, m1, sig1, m2, sig2
    """
    p_est = p
    A_est = A
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2

    for i in range(iter):
        p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_EM_mc(signal_noisy, p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)
        # print("itération {} sur {}".format(i,iter))
        # print({'    p':p_est, 'A':A_est, 'm1':m1_est, 'sig1':sig1_est, 'm2':m2_est, 'sig2':sig2_est})
    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est

def tt_estim_param_EM_gm(iter, signal_noisy, p, m1, sig1, m2, sig2):
    """
    Cette fonction est l'implémentation de l'algorithme EM pour le modèle en question
    :param iter: Nombre d'itération choisie
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: la valeur d'initialisation du vecteur de proba
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de la variance de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de la variance de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme EM donc p, m1, sig1, m2, sig2
    """
    p_est = p
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(iter):
        p_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_EM_gm(signal_noisy, p_est, m1_est, sig1_est, m2_est, sig2_est)
    return p_est, m1_est, sig1_est, m2_est, sig2_est