import numpy as np
from math import log2, sqrt, pi

def gaussienne(x,sig,mu):
    return (1/np.sqrt(2*np.pi*sig**2)*np.exp( -(x-mu)**2/(2*sig**2)) )

def bruit_gauss(signal, w, m1, sig1, m2, sig2):
    """
    Cette fonction permet de bruiter un signal discret à deux classes avec deux gaussiennes
    :param signal: Le signal a bruiter (un numpy array d'int)
    :param w: vecteur dont la première composante est la valeur de la classe 1 et la deuxième est la valeur de la classe 2
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: le signal bruité (numpy array de float)
    """
    output=((signal==w[0])*np.random.normal(m1,sig1,signal.shape)+(signal==w[1])*np.random.normal(m2,sig2,signal.shape))
    # output=np.zeros(signal.shape)
    # for i in range(signal.shape):
    #     if signal[i]==w[0]:
    #         output[i]= np.random.normal(m1,sig1)
    #     else:
    #         output[i]= np.random.normal(m2,sig2)
    return output


def gauss(signal_noisy, m1, sig1, m2, sig2):
    """
    Cette fonction transforme le signal bruité per les densité des deux gaussiennes du bruit
    :param signal_noisy: Le signal bruité (numpy array 1D)
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément de signal noisy
    """
    # output=np.zeros((2,signal_noisy.shape[0]))
    # output[0]=gaussienne(signal_noisy,sig1,m1)
    # output[1]=gaussienne(signal_noisy,sig2,m2)
    # output=np.zeros((signal_noisy.shape[0],2))
    # output[:,0]=gaussienne(signal_noisy,sig1,m1)
    # output[:,1]=gaussienne(signal_noisy,sig2,m2)
    v1=gaussienne(signal_noisy,sig1,m1)
    v2=gaussienne(signal_noisy,sig2,m2)
    output=np.stack((v1,v2),axis=1)
    return output


def calc_erreur(signal1, signal2):
    """
    Cette fonction permet de mesurer la difference entre deux signaux discret (de même taille) à deux classes
    :param signal1: le premier signal, un numpy array
    :param signal2: le deuxième signal, un numpy array
    :return: La différence entre les deux signaux (un float)
    """
    return (np.sum(signal1!=signal2)/np.prod(signal2.shape))


def get_peano_index(dSize):
    """
    Cette fonction permet d'obtenir l'ordre de parcours des pixels d'une image carrée (dont la dimension est une puissance de 2)
    selon la courbe de Hilbert-Peano
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction de fonctionne qu'avec des images carrées)
    :return: une liste de taille 2*dSize*dSize qui correspond aux coordonnées de chaque pixel ordonnée selon le parcours de Hilbert-Peano
    """
    assert log2(dSize).is_integer(), 'veuillez donne une dimension étant une puissance de 2'
    xTmp = 0
    yTmp = 0
    dirTmp = 0
    dirLookup = np.array(
        [[3, 0, 0, 1], [0, 1, 1, 2], [1, 2, 2, 3], [2, 3, 3, 0], [1, 0, 0, 3], [2, 1, 1, 0], [3, 2, 2, 1],
         [0, 3, 3, 2]]).T
    dirLookup = dirLookup + np.array(
        [[4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [0, 4, 4, 0], [0, 4, 4, 0], [0, 4, 4, 0],
         [0, 4, 4, 0]]).T
    orderLookup = np.array(
        [[0, 2, 3, 1], [1, 0, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0], [1, 3, 2, 0], [3, 2, 0, 1], [2, 0, 1, 3],
         [0, 1, 3, 2]]).T
    offsetLookup = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    for i in range(int(log2(dSize))):
        xTmp = np.array([(xTmp - 1) * 2 + offsetLookup[0, orderLookup[0, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[1, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[2, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[3, dirTmp]] + 1])

        yTmp = np.array([(yTmp - 1) * 2 + offsetLookup[1, orderLookup[0, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[1, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[2, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[3, dirTmp]] + 1])

        dirTmp = np.array([dirLookup[0, dirTmp],dirLookup[1, dirTmp], dirLookup[2, dirTmp], dirLookup[3, dirTmp]])

        xTmp = xTmp.T.flatten()
        yTmp = yTmp.T.flatten()
        dirTmp = dirTmp.flatten()

    x = - xTmp
    y = - yTmp
    return x,y


def peano_transform_img(img):
    """
    Cette fonction prend une image carrée (dont la dimension est une puissance de 2) en entrée,
    et retourne l'image applatie (1 dimension) selon le parcours de Hilbert-Peano
    :param img: une image (donc un numpy array 2 dimensions)
    :return: un numpy array 1 dimension
    """
    assert img.shape[0]==img.shape[1], 'veuillez donner une image carrée en entrée'
    assert log2(img.shape[0]).is_integer(), 'veuillez donne rune image dont la dimension est une puissance de 2'
    idx = get_peano_index(img.shape[0])
    return img[idx[0], idx[1]]


def transform_peano_in_img(signal, dSize):
    """
    Cette fonction prend un signal 1D en entrée et une taille, et le transforme en image carrée 2D selon le parcours de Hilbert-Peano
    :param img: un signal 1D
    :return: une image (donc un numpy array 2 dimensions)
    """
    assert dSize==int(sqrt(signal.shape[0])), 'veuillez donner un signal ayant pour dimension dSize^2'
    idx = get_peano_index(dSize)
    img = np.zeros((dSize,dSize))
    img[idx[0], idx[1]] = signal
    return img


def get_line_index(dSize):
    """
    Cette fonction permet d'obtenir l'ordre de parcours des pixels d'une image carrée selon un parcours ligne par ligne
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction de fonctionne qu'avec des images carrées)
    :return: une liste de taille 2*dSize*dSize qui correspond aux coordonnées de chaque pixel ordonnée selon le parcours ligne par ligne
    """
    return [a.flatten() for a in np.indices((dSize, dSize))]


def line_transform_img(img):
    """
    Cette fonction prend une image carrée en entrée, et retourne l'image applatie (1 dimension) selon le parcours ligne par ligne
    :param img: une image (donc un numpy array 2 dimensions)
    :return: un numpy array 1 dimension
    """
    assert img.shape[0]==img.shape[1], 'veuillez donner une image carrée en entrée'
    idx = get_line_index(img.shape[0])
    return img[idx[0], idx[1]]


def transform_line_in_img(signal, dSize):
    """
    Cette fonction prend un signal 1D en entrée et une taille, et le transforme en image carrée 2D selon le parcours ligne par ligne
    :param img: un signal 1D
    :return: une image (donc un numpy array 2 dimensions)
    """
    assert dSize==int(sqrt(signal.shape[0])), 'veuillez donner un signal ayant pour dimension dSize^2'
    idx = get_line_index(dSize)
    img = np.zeros((dSize,dSize))
    img[idx[0], idx[1]] = signal
    return img