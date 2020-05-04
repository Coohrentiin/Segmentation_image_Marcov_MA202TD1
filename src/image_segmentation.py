import numpy as np
import cv2 as cv
from tools import bruit_gauss,calc_erreur, peano_transform_img, transform_peano_in_img, line_transform_img, transform_line_in_img
from gaussian_mixture import *
from markov_chain import *