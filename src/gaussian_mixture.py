import numpy as np
from tools import gauss


def calc_probaprio_gm(signal, w):
    """
    Cete fonction permet de calculer les probabilité a priori des classes w1 et w2, en observant notre signal non bruité
    :param signal: Signal discret non bruité à deux classes (numpy array 1D d'int)
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :return: un vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    """
    p1 = np.sum((signal==w[0]))/signal.shape[0]
    p2 = np.sum((signal == w[1]))/signal.shape[0]
    return np.array([p1,p2])


def mpm_gm(signal_noisy, w, p, m1, sig1, m2, sig2):
    """
    Cette fonction permet d'appliquer la méthode mpm pour retrouver notre signal d'origine à partir de sa version bruité et des paramètres du model.
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    proba_apost = p * gausses
    proba_apost = proba_apost / (proba_apost.sum(axis=1)[..., np.newaxis])
    return w[np.argmax(proba_apost, axis=1)]


def simu_gm(n, w, p):
    """
    Cette fonction permet de simuler un signal discret à 2 classe de taille n àpartir des probabilité d'apparition des deux classes
    :param n: taille du signal
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition pour chaque classe
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    simu = np.zeros((n,), dtype=int)
    for i in range(n):
        aux = np.random.multinomial(1, p)
        simu[i] = w[np.argmax(aux)]
    return simu


def calc_param_EM_gm(signal_noisy, p, m1, sig1, m2, sig2):
    """
    Cette fonction permet de calculer les nouveaux paramètres estimé pour une itération de EM
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: tous les paramètres réestimés donc p, m1, sig1, m2, sig2
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    proba_apost = p * gausses
    proba_apost = proba_apost / (proba_apost.sum(axis=1)[..., np.newaxis])
    p = proba_apost.sum(axis=0)/proba_apost.shape[0]
    m1 = (proba_apost[:,0] * signal_noisy).sum()/proba_apost[:,0].sum()
    sig1 = np.sqrt((proba_apost[:,0]*((signal_noisy-m1)**2)).sum()/proba_apost[:,0].sum())
    m2 = (proba_apost[:, 1] * signal_noisy).sum() / proba_apost[:, 1].sum()
    sig2 = np.sqrt((proba_apost[:, 1] * ((signal_noisy - m2) ** 2)).sum() / proba_apost[:, 1].sum())
    return p, m1, sig1, m2, sig2


def estim_param_EM_gm(iter, signal_noisy, p, m1, sig1, m2, sig2):
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
        print({'p':p_est, 'm1':m1_est, 'sig1':sig1_est, 'm2':m2_est, 'sig2':sig2_est})
    return p_est, m1_est, sig1_est, m2_est, sig2_est
