import numpy as np
from tools import gauss

def forward(A, p, gauss):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs forward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: un vecteur de taille la longueur de la chaîne, contenant tous les forward (de 1 à n)
    """
    n=len(gauss)
    alpha=np.zeros( (n,2) )

    alpha[0]=p*gauss[0] #valeur initiale p(x1,y1)
    alpha[0]=alpha[0]/(alpha[0].sum())

    for i in range(1,n):
        alpha[i]=(alpha[i-1]@A)*gauss[i]
        alpha[i]=alpha[i]/(alpha[i].sum())
    return alpha


def backward(A, gauss):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs backward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: un vecteur de taille la longueur de la chaîne, contenant tous les backward (de 1 à n).
    Attention, si on calcule les backward en partant de la fin de la chaine, je conseille quand même d'ordonner le vecteur backward du début à la fin
    """
    # n=gauss.shape[1]
    n=len(gauss)
    beta=np.zeros( (n,2) )
    beta[n-1]=np.ones(2)

    for i in range(n-2,-1,-1):
        beta[i]= A@(beta[i+1]*gauss[i+1])
        beta[i]=beta[i]/( beta[i].sum())
    return beta


def mpm_mc(signal_noisy, w, p, A, m1, sig1, m2, sig2):
    """
     Cette fonction permet d'appliquer la méthode mpm pour retrouver notre signal d'origine à partir de sa version bruité et des paramètres du model.
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    alpha = forward(A,p,gausses)
    beta = backward(A,gausses)
    # print("alpha:")
    # print(alpha)
    # print("bata:")
    # print(beta)
    posteriori=(alpha * beta)/ ( (alpha*beta).sum(axis=1)[...,np.newaxis])
    # posteriori=(alpha * beta)
    # print("postério:")
    # print(posteriori)
    mpm= w[np.argmax(posteriori,axis=1)]    
    return mpm


def calc_probaprio_mc(signal, w):
    """
    Cete fonction permet de calculer les probabilité a priori des classes w1 et w2 et les transitions a priori d'une classe à l'autre,
    en observant notre signal non bruité
    :param signal: Signal discret non bruité à deux classes (numpy array 1D d'int)
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :return: un vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    """
    p1=np.sum((signal==w[0]))/signal.shape[0]
    p2=np.sum((signal==w[1]))/signal.shape[0]
    p=np.array([p1,p2])
    c=sum( [ np.array([[(signal[i]==k and signal[i+1]==l) for l in w] for k in w]) for i in range(signal.shape[0]-1)])/signal.shape[0]
    A=(c.T/p).T
    return p, A


def simu_mc(n, w, p, A):
    """
    Cette fonction permet de simuler un signal discret à 2 classe de taille n à partir des probabilité d'apparition des deux classes et de la Matrice de transition
    :param n: taille du signal
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    simu = np.zeros((n,), dtype=int)
    aux = np.random.multinomial(1, p)
    simu[0] = w[np.argmax(aux)]
    for i in range(1, n):
        aux = np.random.multinomial(1, A[np.where(w == simu[i - 1])[0][0], :])
        simu[i] = w[np.argmax(aux)]
    return simu


def calc_param_EM_mc(signal_noisy, p, A, m1, sig1, m2, sig2):
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


def estim_param_EM_mc(iter, signal_noisy, p, A, m1, sig1, m2, sig2):
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
        print("itération {} sur {}".format(i,iter))
        print({'    p':p_est, 'A':A_est, 'm1':m1_est, 'sig1':sig1_est, 'm2':m2_est, 'sig2':sig2_est})
    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est

